from executor import trainer, tester

def execute(args) :     
    if args['executor']['type'] == 'trainer' : 
        print('훈련모드 진입')
        trainer.trainer(args).train()
    elif args['executor']['type'] == 'tester' : 
        print('테스트모드 진입')
        tester.tester(args).test()
    else : 
        print('오류. 인자를 다시 확인하십시오.')