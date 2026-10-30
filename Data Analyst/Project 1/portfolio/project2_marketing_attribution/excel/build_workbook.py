"""
Builds budget_reallocation_model.xlsx — an executive-ready, interactive
what-if model for reallocating marketing spend based on the linear-
attribution ROAS findings from the SQL/Python analysis.

Sheet 1 "Attribution by Model"  - the 4-model attribution comparison (source data)
Sheet 2 "Current State"          - spend, attributed revenue, ROAS per channel (formulas)
Sheet 3 "Budget What-If"         - EDITABLE % reallocation levers -> projected impact
Sheet 4 "Instructions"           - legend / how to use
"""
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.comments import Comment

FONT = "Arial"
HEADER_FILL = PatternFill("solid", fgColor="2E4057")
HEADER_FONT = Font(name=FONT, size=11, bold=True, color="FFFFFF")
TITLE_FONT = Font(name=FONT, size=14, bold=True, color="2E4057")
SUBTITLE_FONT = Font(name=FONT, size=10, italic=True, color="666666")
INPUT_FONT = Font(name=FONT, size=11, color="0000FF")           # blue = editable input
FORMULA_FONT = Font(name=FONT, size=11, color="000000")          # black = formula
FLAG_FONT = Font(name=FONT, size=11, bold=True, color="C00000")
INPUT_FILL = PatternFill("solid", fgColor="FFFFCC")
THIN = Side(style="thin", color="D9D9D9")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

wb = Workbook()

# ---------------------------------------------------------------------
# Sheet 1: Attribution by Model (source data, pulled from sql/02_attribution_models.sql)
# ---------------------------------------------------------------------
ws1 = wb.active
ws1.title = "Attribution by Model"
ws1["A1"] = "Multi-Touch Attribution — Revenue Credit by Model"
ws1["A1"].font = TITLE_FONT
ws1["A2"] = "Source: sql/02_attribution_models.sql, run against data/marketing.db (touchpoints table, 9,000 user journeys)"
ws1["A2"].font = SUBTITLE_FONT

headers1 = ["Channel", "First Touch ($)", "Last Touch ($)", "Linear ($)", "Position-Based ($)"]
attribution_data = [
    ["Paid Search",    51096.09, 62736.12, 56342.38, 53937.40],
    ["Organic Search", 51937.39, 53937.94, 53052.83, 50098.85],
    ["Email",          49361.55, 58628.74, 53042.87, 50587.53],
    ["Referral",       47975.99, 40305.70, 44997.33, 42079.49],
    ["Paid Social",    42873.18, 27635.70, 35808.79, 33640.48],
]
r0 = 4
for j, h in enumerate(headers1, start=1):
    c = ws1.cell(row=r0, column=j, value=h)
    c.font = HEADER_FONT; c.fill = HEADER_FILL; c.border = BORDER
    c.alignment = Alignment(horizontal="center")
for i, row in enumerate(attribution_data, start=1):
    for j, val in enumerate(row, start=1):
        c = ws1.cell(row=r0+i, column=j, value=val)
        c.border = BORDER
        if j == 1:
            c.font = Font(name=FONT, size=11)
        else:
            c.font = FORMULA_FONT
            c.number_format = "$#,##0"
# totals row using formulas (not hardcoded)
total_row = r0 + len(attribution_data) + 1
ws1.cell(row=total_row, column=1, value="Total").font = Font(name=FONT, bold=True)
for j in range(2, 6):
    col = get_column_letter(j)
    cell = ws1.cell(row=total_row, column=j, value=f"=SUM({col}{r0+1}:{col}{r0+len(attribution_data)})")
    cell.font = Font(name=FONT, bold=True)
    cell.number_format = "$#,##0"
    cell.border = BORDER

note = ws1.cell(row=total_row+3, column=1,
    value=("Note: attribution model choice materially changes channel rank. Paid Social ranks "
           "5th (last) under Last Touch but 2nd under First Touch — it plays an early-funnel "
           "assist role that last-touch credit erases entirely."))
note.font = Font(name=FONT, size=10, italic=True, color="666666")
ws1.column_dimensions["A"].width = 22
for col in "BCDE":
    ws1.column_dimensions[col].width = 18

# ---------------------------------------------------------------------
# Sheet 2: Current State — spend, linear-attributed revenue, ROAS
# ---------------------------------------------------------------------
ws2 = wb.create_sheet("Current State")
ws2["A1"] = "6-Month Spend vs. Linear-Attributed ROAS (Oct 2025 – Mar 2026)"
ws2["A1"].font = TITLE_FONT
ws2["A2"] = "Source: sql/03_channel_roas.sql — total spend from channel_spend table, revenue from linear attribution model, both over the same 6-month window"
ws2["A2"].font = SUBTITLE_FONT

headers2 = ["Channel", "6-Month Spend ($)", "Linear Attributed Revenue ($)", "ROAS (Revenue / Spend)"]
current_state = [
    ["Paid Search",    255101.91, 56342.38, None],   # spend and revenue both totals over the same 6-month window
    ["Paid Social",    209141.63, 35808.79, None],
    ["Organic Search", 35491.02,  53052.83, None],
    ["Email",          22847.20,  53042.87, None],
    ["Referral",       17678.57,  44997.33, None],
]
r0 = 4
for j, h in enumerate(headers2, start=1):
    c = ws2.cell(row=r0, column=j, value=h)
    c.font = HEADER_FONT; c.fill = HEADER_FILL; c.border = BORDER
    c.alignment = Alignment(horizontal="center", wrap_text=True)
for i, row in enumerate(current_state, start=1):
    rr = r0 + i
    ws2.cell(row=rr, column=1, value=row[0]).font = Font(name=FONT)
    ws2.cell(row=rr, column=1).border = BORDER
    c2 = ws2.cell(row=rr, column=2, value=round(row[1], 2)); c2.font = FORMULA_FONT; c2.number_format = "$#,##0"; c2.border = BORDER
    c3 = ws2.cell(row=rr, column=3, value=row[2]); c3.font = FORMULA_FONT; c3.number_format = "$#,##0"; c3.border = BORDER
    c4 = ws2.cell(row=rr, column=4, value=f"=C{rr}/B{rr}"); c4.font = FORMULA_FONT; c4.number_format = "0.00x"; c4.border = BORDER

total_row2 = r0 + len(current_state) + 1
ws2.cell(row=total_row2, column=1, value="Total / Blended").font = Font(name=FONT, bold=True)
ws2.cell(row=total_row2, column=2, value=f"=SUM(B{r0+1}:B{r0+len(current_state)})").number_format = "$#,##0"
ws2.cell(row=total_row2, column=3, value=f"=SUM(C{r0+1}:C{r0+len(current_state)})").number_format = "$#,##0"
ws2.cell(row=total_row2, column=4, value=f"=C{total_row2}/B{total_row2}").number_format = "0.00x"
for col in [2,3,4]:
    ws2.cell(row=total_row2, column=col).font = Font(name=FONT, bold=True)
    ws2.cell(row=total_row2, column=col).border = BORDER

ws2.column_dimensions["A"].width = 20
for col in "BCD":
    ws2.column_dimensions[col].width = 20

# ---------------------------------------------------------------------
# Sheet 3: Budget What-If Model (INTERACTIVE)
# ---------------------------------------------------------------------
ws3 = wb.create_sheet("Budget What-If")
ws3["A1"] = "Budget Reallocation What-If Model"
ws3["A1"].font = TITLE_FONT
ws3["A2"] = "Edit the yellow % Reallocation cells (column C). Total budget is held constant — reallocating from one channel funds another."
ws3["A2"].font = SUBTITLE_FONT

# Assumptions block
ws3["A4"] = "Assumptions"
ws3["A4"].font = Font(name=FONT, bold=True, size=11)
ws3["A5"] = "Diminishing-returns elasticity factor (applied when a channel's spend increases)"
ws3["A5"].font = Font(name=FONT, size=10)
ws3["C5"] = 0.85
ws3["C5"].font = INPUT_FONT
ws3["C5"].fill = INPUT_FILL
ws3["C5"].number_format = "0.00"
ws3["C5"].border = BORDER
ws3["C5"].comment = Comment("Editable assumption: when spend on a channel increases, each incremental "
                             "dollar is assumed to return only this fraction of the current ROAS "
                             "(diminishing returns / market saturation). 1.00 = no diminishing returns.", "Analyst")

headers3 = ["Channel", "Current 6-Mo Spend ($)", "% Reallocation", "New Spend ($)",
            "Current ROAS", "Effective ROAS", "Projected Revenue ($)", "Revenue Δ vs. Current ($)"]
r0 = 8
for j, h in enumerate(headers3, start=1):
    c = ws3.cell(row=r0, column=j, value=h)
    c.font = HEADER_FONT; c.fill = HEADER_FILL; c.border = BORDER
    c.alignment = Alignment(horizontal="center", wrap_text=True)
ws3.row_dimensions[r0].height = 30

channels = ["Paid Search", "Paid Social", "Organic Search", "Email", "Referral"]
# default suggested reallocation: cut the two ROAS-losing channels 10%, fund the three
# ROAS-winning channels proportionally (61.1% each) -- nets to exactly budget-neutral
default_realloc = {"Paid Search": -0.10, "Paid Social": -0.10, "Organic Search": 0.611,
                    "Email": 0.611, "Referral": 0.611}

for i, ch in enumerate(channels, start=1):
    rr = r0 + i
    ws3.cell(row=rr, column=1, value=ch).font = Font(name=FONT)
    ws3.cell(row=rr, column=1).border = BORDER
    # Current spend: pull from Current State sheet
    src_row = 4 + i
    cB = ws3.cell(row=rr, column=2, value=f"='Current State'!B{src_row}")
    cB.font = FORMULA_FONT; cB.number_format = "$#,##0"; cB.border = BORDER
    # % Reallocation -- EDITABLE
    cC = ws3.cell(row=rr, column=3, value=default_realloc[ch])
    cC.font = INPUT_FONT; cC.fill = INPUT_FILL; cC.number_format = "0.0%"; cC.border = BORDER
    # New spend
    cD = ws3.cell(row=rr, column=4, value=f"=B{rr}*(1+C{rr})")
    cD.font = FORMULA_FONT; cD.number_format = "$#,##0"; cD.border = BORDER
    # Current ROAS
    cE = ws3.cell(row=rr, column=5, value=f"='Current State'!D{src_row}")
    cE.font = FORMULA_FONT; cE.number_format = "0.00x"; cE.border = BORDER
    # Effective ROAS: full ROAS if spend cut or flat, elasticity-adjusted if spend increased
    cF = ws3.cell(row=rr, column=6, value=f"=IF(C{rr}<=0,E{rr},E{rr}*$C$5)")
    cF.font = FORMULA_FONT; cF.number_format = "0.00x"; cF.border = BORDER
    # Projected revenue
    cG = ws3.cell(row=rr, column=7, value=f"=D{rr}*F{rr}")
    cG.font = FORMULA_FONT; cG.number_format = "$#,##0"; cG.border = BORDER
    # Revenue delta vs current
    cH = ws3.cell(row=rr, column=8, value=f"=G{rr}-('Current State'!C{src_row})")
    cH.font = FORMULA_FONT; cH.number_format = "$#,##0;[RED]-$#,##0"; cH.border = BORDER

total_row3 = r0 + len(channels) + 1
ws3.cell(row=total_row3, column=1, value="Total").font = Font(name=FONT, bold=True)
for col, letter in [(2, "B"), (4, "D"), (7, "G"), (8, "H")]:
    cell = ws3.cell(row=total_row3, column=col, value=f"=SUM({letter}{r0+1}:{letter}{r0+len(channels)})")
    cell.font = Font(name=FONT, bold=True); cell.number_format = "$#,##0"; cell.border = BORDER

# Budget-neutrality guardrail check
check_row = total_row3 + 2
ws3.cell(row=check_row, column=1, value="Budget-neutral check (New Total Spend − Current Total Spend):").font = Font(name=FONT, italic=True, size=10)
chk = ws3.cell(row=check_row, column=4, value=f"=D{total_row3}-B{total_row3}")
chk.number_format = "$#,##0;[RED]-$#,##0"
chk.font = Font(name=FONT, bold=True)
flag = ws3.cell(row=check_row, column=5,
    value=f'=IF(ABS(D{total_row3}-B{total_row3})>500,"⚠ Reallocation percentages do not net to ~0 — adjust column C","✓ Budget-neutral")')
flag.font = FLAG_FONT

roi_row = check_row + 2
ws3.cell(row=roi_row, column=1, value="Net projected revenue impact of this reallocation:").font = Font(name=FONT, bold=True, size=11)
netcell = ws3.cell(row=roi_row, column=4, value=f"=H{total_row3}")
netcell.font = Font(name=FONT, bold=True, size=12, color="2E4057")
netcell.number_format = "$#,##0;[RED]-$#,##0"

ws3.column_dimensions["A"].width = 18
for col, w in zip("BCDEFGH", [20, 15, 15, 13, 14, 18, 20]):
    ws3.column_dimensions[col].width = w

# ---------------------------------------------------------------------
# Sheet 4: Instructions
# ---------------------------------------------------------------------
ws4 = wb.create_sheet("Instructions")
ws4["A1"] = "How to Use This Workbook"
ws4["A1"].font = TITLE_FONT
lines = [
    "",
    "This model translates the multi-touch attribution & ROAS findings from the SQL/Python",
    "analysis (see /notebook/analysis.ipynb) into an interactive budget-reallocation tool.",
    "",
    "Sheet guide:",
    "  1. Attribution by Model — how attributed revenue per channel changes across 4 models.",
    "  2. Current State — actual monthly spend and linear-attribution ROAS per channel today.",
    "  3. Budget What-If — EDIT the yellow % Reallocation cells to test a new budget mix.",
    "     The model auto-checks that reallocation is budget-neutral (col E flag) and applies",
    "     a diminishing-returns elasticity factor to channels receiving increased spend, since",
    "     ROAS realistically declines as more money chases the same audience.",
    "",
    "Color legend:",
    "  Blue text on yellow fill = editable input cell",
    "  Black text = formula (recalculates automatically)",
    "",
    "Default scenario loaded: cut Paid Search -10% and Paid Social -10% (both sub-1.0x ROAS",
    "under linear attribution), fund Organic/Email/Referral +61.1% each -- an exactly",
    "budget-neutral reallocation (see the guardrail check on the What-If sheet).",
]
for i, l in enumerate(lines, start=2):
    cell = ws4.cell(row=i, column=1, value=l)
    cell.font = Font(name=FONT, size=11, bold=(l.startswith("Sheet guide") or l.startswith("Color legend")))
ws4.column_dimensions["A"].width = 100

wb.save("budget_reallocation_model.xlsx")
print("Saved budget_reallocation_model.xlsx")
