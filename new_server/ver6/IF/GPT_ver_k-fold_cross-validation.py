from sklearn.model_selection import KFold

# k-Fold 설정
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

# Dataset 전체 가져오기
full_dataset = MITLoader_MLP_binary(csv_file=train_path)

# k-Fold 수행
for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
    print(f"Starting fold {fold + 1}/{k_folds}...")

    # Train/Validation 데이터셋 분리
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # TensorBoard 폴더 설정
    writer = SummaryWriter(log_dir=f"./tensorboard/{model_name}_fold{fold + 1}")

    # SNN 네트워크 초기화
    model = SNN_MLP(num_encoders=num_encoders, num_classes=num_classes, hidden_size=hidden_size,
                    hidden_size_2=hidden_size_2, threshold_value=threshold_value, bias_option=need_bias).to(device)

    encoder = encoding.PoissonEncoder()

    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(train_params, lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=scheduler_tmax, eta_min=scheduler_eta_min)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        accuracy.reset()
        f1_micro.reset()
        f1_weighted.reset()
        auroc_macro.reset()
        auroc_weighted.reset()
        auprc.reset()

        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device=device).squeeze(1)
            targets = targets.to(device=device)
            label_onehot = F.one_hot(targets, num_classes).float()

            out_fr = 0.
            for t in range(timestep):
                encoded_data = encoder(data)
                out_fr += model(encoded_data)

            out_fr /= timestep
            loss = F.mse_loss(out_fr, label_onehot, reduction='none')
            weighted_loss = loss * class_weight[targets].unsqueeze(1)
            final_loss = weighted_loss.mean()

            total_loss += final_loss.item()
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            preds = torch.argmax(out_fr, dim=1)
            accuracy.update(preds, targets)
            f1_micro.update(preds, targets)
            f1_weighted.update(preds, targets)
            auroc_macro.update(preds, targets)
            auroc_weighted.update(preds, targets)
            probabilities = F.softmax(out_fr, dim=1)[:, 1]
            auprc.update(probabilities, targets)
            functional.reset_net(model)

        scheduler.step()

        # Train Metrics Logging
        train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy.compute()
        writer.add_scalar('train_Loss', train_loss, epoch)
        writer.add_scalar('train_Accuracy', train_accuracy, epoch)

        # Validation
        valid_loss = check_accuracy(val_loader, model)

        print(f'Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Valid Loss: {valid_loss}')

    # Fold-specific TensorBoard Writer Close
    writer.close()
