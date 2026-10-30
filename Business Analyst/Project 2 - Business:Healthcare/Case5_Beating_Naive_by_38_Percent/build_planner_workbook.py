"""
Live Excel workbook: turns the Holt-Winters forecast + backtested error into
a weekly sales-planning budget with a statistically-grounded safety buffer.
Follows the xlsx skill's financial-model conventions.
"""
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.comments import Comment
import pandas as pd

wb = Workbook()
ws = wb.active
ws.title = "Sales Planning"

FONT = "Arial"
BLUE = Font(name=FONT, color="0000FF")
BLACK = Font(name=FONT, color="000000")
BOLD = Font(name=FONT, bold=True)
TITLE_FONT = Font(name=FONT, bold=True, size=14)
HEADER_FONT = Font(name=FONT, bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="1E2530")
YELLOW = PatternFill("solid", fgColor="FFFF00")
SECTION_FILL = PatternFill("solid", fgColor="EFEAE0")
thin = Side(style="thin", color="DDD6C7")
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)

ws.column_dimensions["A"].width = 40
for col in "BCDEFG":
    ws.column_dimensions[col].width = 16

def section(row, text, span=6):
    ws.cell(row=row, column=1, value=text).font = BOLD
    for col in range(1, span+1):
        ws.cell(row=row, column=col).fill = SECTION_FILL
    return row + 1

ws.merge_cells("A1:F1")
ws["A1"] = "Case 05 — Store 20 Weekly Sales Planning & Safety-Buffer Workbook"
ws["A1"].font = TITLE_FONT
ws.merge_cells("A2:F2")
ws["A2"] = "Forecast: Holt-Winters (additive trend + 52-week seasonal), fit on 143 weeks of real Walmart sales data"
ws["A2"].font = Font(name=FONT, italic=True, size=9, color="52514E")

r = 4
r = section(r, "1. NEXT 4 WEEKS — HOLT-WINTERS FORECAST  (blue = model output)")
fc = pd.read_csv("next_4_week_forecast.csv")
fc.columns = ["Week", "Forecast"]
header_row = r
ws.cell(row=r, column=1, value="Week starting").font = HEADER_FONT
ws.cell(row=r, column=1).fill = HEADER_FILL
ws.cell(row=r, column=2, value="Forecasted sales ($)").font = HEADER_FONT
ws.cell(row=r, column=2).fill = HEADER_FILL
r += 1
first_forecast_row = r
for _, row_data in fc.iterrows():
    ws.cell(row=r, column=1, value=str(row_data["Week"])[:10]).font = BLUE
    c = ws.cell(row=r, column=2, value=round(float(row_data["Forecast"]), 0))
    c.font = BLUE
    c.number_format = "$#,##0"
    c.border = BORDER
    r += 1
last_forecast_row = r - 1
r += 1

r = section(r, "2. BACKTESTED FORECAST ACCURACY  (blue = from the 16-week holdout test)")
row_error_std = r
ws.cell(row=r, column=1, value="Std. dev. of forecast error, $ (16-week backtest)").font = BLACK
c = ws.cell(row=r, column=2, value=53580)
c.font = BLUE; c.number_format = "$#,##0"; c.border = BORDER
c.comment = Comment("From run_analysis.py backtest: std dev of |Actual - Holt-Winters forecast| over the 16-week holdout period.", "BA Portfolio Playbook")
r += 1
row_mape = r
ws.cell(row=r, column=1, value="Backtested MAPE").font = BLACK
c = ws.cell(row=r, column=2, value=0.0237)
c.font = BLUE; c.number_format = "0.0%"; c.border = BORDER
r += 1

r = section(r, "3. PLANNING ASSUMPTIONS  (yellow = edit for your own service level / lead time)")
row_lead_time = r
ws.cell(row=r, column=1, value="Planning lead time (weeks)").font = BLACK
c = ws.cell(row=r, column=2, value=1)
c.font = BLUE; c.fill = YELLOW; c.border = BORDER
c.comment = Comment("How many weeks ahead this budget/staffing decision is locked in. Increase if your team commits schedules further in advance.", "BA Portfolio Playbook")
r += 1
row_service_level_label = r
ws.cell(row=r, column=1, value="Desired service level").font = BLACK
c = ws.cell(row=r, column=2, value="95%")
c.font = BLUE; c.fill = YELLOW; c.border = BORDER
c.comment = Comment("Must match a value in the z-score reference table (90%, 95%, 97.5%, 99%). Higher service level = bigger safety buffer, less risk of under-planning.", "BA Portfolio Playbook")
r += 2

r = section(r, "Z-SCORE REFERENCE TABLE")
z_table_header = r
ws.cell(row=r, column=1, value="Service level").font = HEADER_FONT
ws.cell(row=r, column=1).fill = HEADER_FILL
ws.cell(row=r, column=2, value="Z-score").font = HEADER_FONT
ws.cell(row=r, column=2).fill = HEADER_FILL
r += 1
z_table_first = r
z_values = [("90%", 1.28), ("95%", 1.65), ("97.5%", 1.96), ("99%", 2.33)]
for lvl, z in z_values:
    ws.cell(row=r, column=1, value=lvl).font = BLACK
    c = ws.cell(row=r, column=2, value=z); c.font = BLACK; c.border = BORDER
    r += 1
z_table_last = r - 1
r += 1

r = section(r, "4. SAFETY BUFFER & PLANNING BUDGET  (formulas — recalculate automatically)")
row_z_lookup = r
ws.cell(row=r, column=1, value="Z-score for chosen service level").font = BLACK
c = ws.cell(row=r, column=2,
            value=f"=INDEX(B{z_table_first}:B{z_table_last},MATCH(B{row_service_level_label},A{z_table_first}:A{z_table_last},0))")
c.font = BLACK; c.number_format = "0.00"; c.border = BORDER
row_safety_buffer = r + 1
ws.cell(row=row_safety_buffer, column=1, value="Safety buffer ($) = Z × error std dev × √(lead time)").font = BLACK
c = ws.cell(row=row_safety_buffer, column=2, value=f"=B{row_z_lookup}*B{row_error_std}*SQRT(B{row_lead_time})")
c.font = BLACK; c.number_format = "$#,##0"; c.border = BORDER
r = row_safety_buffer + 2

r = section(r, "5. PLANNED BUDGET PER WEEK  (forecast + safety buffer)")
plan_header = r
ws.cell(row=r, column=1, value="Week starting").font = HEADER_FONT
ws.cell(row=r, column=1).fill = HEADER_FILL
ws.cell(row=r, column=2, value="Forecast ($)").font = HEADER_FONT
ws.cell(row=r, column=2).fill = HEADER_FILL
ws.cell(row=r, column=3, value="+ Safety buffer").font = HEADER_FONT
ws.cell(row=r, column=3).fill = HEADER_FILL
ws.cell(row=r, column=4, value="Planned budget").font = HEADER_FONT
ws.cell(row=r, column=4).fill = HEADER_FILL
r += 1
plan_first = r
for i, fr in enumerate(range(first_forecast_row, last_forecast_row + 1)):
    ws.cell(row=r, column=1, value=f"=A{fr}").font = BLACK
    ws.cell(row=r, column=2, value=f"=B{fr}").font = BLACK
    ws.cell(row=r, column=2).number_format = "$#,##0"
    ws.cell(row=r, column=3, value=f"=$B${row_safety_buffer}").font = BLACK
    ws.cell(row=r, column=3).number_format = "$#,##0"
    c = ws.cell(row=r, column=4, value=f"=B{r}+C{r}")
    c.font = BOLD; c.number_format = "$#,##0"; c.border = BORDER
    r += 1
plan_last = r - 1
ws.cell(row=r, column=1, value="4-week total planned budget").font = BOLD
c = ws.cell(row=r, column=4, value=f"=SUM(D{plan_first}:D{plan_last})")
c.font = BOLD; c.number_format = "$#,##0"; c.border = BORDER
r += 2

r = section(r, "LEGEND")
ws.cell(row=r, column=1, value="Blue text").font = BLACK
ws.cell(row=r, column=2, value="Model output or editable input").font = BLACK
r += 1
ws.cell(row=r, column=1, value="Yellow fill").font = BLACK
ws.cell(row=r, column=2, value="Key assumption — edit for your own planning").font = BLACK
r += 1
ws.cell(row=r, column=1, value="Black text").font = BLACK
ws.cell(row=r, column=2, value="Formula — recalculates automatically").font = BLACK

wb.save("case5_sales_planning_workbook.xlsx")
print("Workbook saved.")
