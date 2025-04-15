import svgutils.transform as sg
from lxml import etree
import cairosvg

# 입력 SVG 파일과 라벨 정의
svg_files = [
    ("T-SNE_poisson.svg", "(a)"),
    ("T-SNE_burst.svg", "(b)"),
    ("T-SNE_filterCNN.svg", "(c)")
]

# SVG 레이아웃 설정
PLOT_WIDTH = 500
PLOT_HEIGHT = 400
SPACING = 40

elements = []

for idx, (svg_path, label_text) in enumerate(svg_files):
    fig = sg.fromfile(svg_path)
    plot = fig.getroot()
    x_offset = idx * (PLOT_WIDTH + SPACING)
    plot.moveto(x_offset, 0)

    # (a)/(b)/(c) 라벨 삽입
    label = sg.TextElement(
        x_offset - 25, 25,
        label_text,
        size=30,
        font="Arial",
        weight="bold"
    )

    elements.extend([plot, label])

# 캔버스 크기 계산
total_width = len(svg_files) * (PLOT_WIDTH + SPACING)
canvas = sg.SVGFigure(f"{total_width}px", f"{PLOT_HEIGHT}px")
canvas.append(elements)

# 1차 SVG 저장
svg_output = "merged_labeled_plots.svg"
canvas.save(svg_output)

# <svg> 루트에 width/height 속성 강제 삽입
parser = etree.XMLParser()
tree = etree.parse(svg_output, parser)
root = tree.getroot()
root.set("width", f"{total_width}px")
root.set("height", f"{PLOT_HEIGHT}px")

# 덮어쓰기 저장
tree.write(svg_output)

# PDF 변환
pdf_output = "merged_labeled_plots.pdf"
cairosvg.svg2pdf(url=svg_output, write_to=pdf_output)

print("✅ 병합 및 PDF 저장 완료!")