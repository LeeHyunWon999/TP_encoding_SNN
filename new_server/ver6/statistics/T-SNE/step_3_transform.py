from lxml import etree
import cairosvg

# 플롯 관련 설정 (사용한 병합 스크립트 기준)
PLOT_WIDTH = 500
PLOT_HEIGHT = 400
SPACING = 40
NUM_PLOTS = 3

# 여백 설정 (단위: px)
LEFT_MARGIN = 60
RIGHT_MARGIN = 20
TOP_MARGIN = 30
BOTTOM_MARGIN = 40

# 최종 캔버스 크기 계산
total_width = NUM_PLOTS * (PLOT_WIDTH + SPACING) - SPACING + LEFT_MARGIN + RIGHT_MARGIN
total_height = PLOT_HEIGHT + TOP_MARGIN + BOTTOM_MARGIN

# 기존 SVG 로드
tree = etree.parse("merged_labeled_plots.svg")
root = tree.getroot()

# 루트에 정확한 크기 삽입
root.set("width", f"{total_width}px")
root.set("height", f"{total_height}px")

# 내부 요소 전체를 <g transform="translate(...)">로 감싸기
from copy import deepcopy
original_children = list(root)
for child in original_children:
    root.remove(child)

# 새 그룹 만들기
group = etree.Element("g")
group.set("transform", f"translate({LEFT_MARGIN},0)")

# 원래 자식들을 그룹 안에 넣음
for child in original_children:
    group.append(deepcopy(child))

# 그룹을 루트에 삽입
root.append(group)

# 저장
fixed_svg = "merged_labeled_plots_fixed.svg"
tree.write(fixed_svg)
print(f"✅ SVG 크기 및 위치 재설정 완료: {fixed_svg} ({total_width}px x {total_height}px)")

# PDF 변환
output_pdf = "merged_labeled_plots.pdf"
cairosvg.svg2pdf(url=fixed_svg, write_to=output_pdf)
print(f"✅ PDF 생성 완료: {output_pdf}")
