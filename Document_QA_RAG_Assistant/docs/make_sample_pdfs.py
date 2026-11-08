from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, spaceAfter=14)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=6)
BODY = ParagraphStyle("Body", parent=styles["Normal"], fontSize=11, leading=16, spaceAfter=8)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=14, bulletIndent=2)


def build(path, footer_title, sections):
    """Each section is one page: (heading, [("p"|"h"|"b", text), ...])."""

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(2 * cm, 1.2 * cm, f"{footer_title} | Page {doc.page}")
        canvas.restoreState()

    story = []
    for i, (heading, blocks) in enumerate(sections):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(heading, H1))
        for kind, text in blocks:
            if kind == "h":
                story.append(Paragraph(text, H2))
            elif kind == "b":
                story.append(Paragraph(text, BULLET, bulletText="\u2022"))
            else:
                story.append(Paragraph(text, BODY))
        story.append(Spacer(1, 6))
    SimpleDocTemplate(path, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                      topMargin=2.2 * cm, bottomMargin=2.2 * cm,
                      title=footer_title, author="Fictional sample document").build(
        story, onFirstPage=footer, onLaterPages=footer)


HANDBOOK = [
    ("Welcome to Aurelia Robotics", [
        ("p", "Aurelia Robotics Ltd. was founded in 2014 by Dr. Mirela Voss and Tomas Eklund in Kestrel Bay. "
              "Today we employ about 340 people across three sites: our headquarters in Kestrel Bay, the "
              "manufacturing plant in Port Alder, and the research office in Lindenmoor."),
        ("p", "We build warehouse automation robots, including the Heron picking arm and the Marlin pallet shuttle. "
              "Our customers are logistics companies in more than twenty countries."),
        ("h", "Our values"),
        ("p", "Three values guide how we work: build carefully, ship honestly, and learn loudly. We expect every "
              "employee to raise problems early and to share what they learn with their team."),
        ("h", "About this handbook"),
        ("p", "This handbook applies to all employees on permanent and fixed-term contracts. It was last revised in "
              "January 2026 by the People Team, led by Head of People Ingrid Halloran. Questions can be sent to "
              "people@aurelia-robotics.example."),
    ]),
    ("Working Hours and Remote Work", [
        ("b", "The standard working week is 38 hours. Core hours are 10:00 to 15:00, during which employees should "
              "be reachable by their team."),
        ("b", "Hybrid work: employees in office-based roles are expected to be in the office at least 2 days per week. "
              "Manufacturing roles at Port Alder are on-site and follow published shift plans."),
        ("b", "Remote work abroad is allowed for up to 20 working days per calendar year, with prior approval from "
              "the line manager. A single stay abroad may not exceed 10 consecutive working days, and the People "
              "Team must be told the destination country at least 14 days in advance."),
        ("b", "Home office equipment: every employee receives a one-time allowance of EUR 450 for a desk and chair, "
              "plus a monthly internet stipend of EUR 25."),
        ("b", "Overtime is not paid out. It is compensated with time off in lieu, which must be taken within "
              "3 months."),
    ]),
    ("Leave and Time Off", [
        ("b", "Annual leave: full-time employees receive 27 paid days per year, plus 1 additional day for every "
              "2 full years of service, up to a maximum of 32 days."),
        ("b", "Carry-over: up to 5 unused leave days can be carried into the next year and must be used by 31 March."),
        ("b", "Sick leave is paid in full. A doctor's note is only required after 3 consecutive days of absence."),
        ("b", "Parental leave: the primary caregiver receives 18 weeks at full pay, and the secondary caregiver "
              "receives 8 weeks at full pay."),
        ("b", "Volunteer days: every employee gets 2 paid volunteer days per year to work for a charity of their choice."),
        ("b", "Public holidays follow the official calendar of the region where the employee's site is located."),
    ]),
    ("Travel and Expenses", [
        ("b", "Meal allowance: up to EUR 45 per day for domestic business travel and EUR 70 per day for "
              "international business travel."),
        ("b", "Flights: economy class for flights shorter than 6 hours; premium economy is allowed for flights of "
              "6 hours or longer. Business class is not reimbursed."),
        ("b", "Hotels: up to EUR 140 per night, or EUR 190 per night in high-cost cities such as London and Zurich."),
        ("b", "Receipts must be submitted through the Expense Portal within 30 days. Any single expense above "
              "EUR 500 needs prior approval from a director."),
        ("b", "Learning budget: every employee has EUR 1,800 per year for courses, books and conferences. Unused "
              "budget does not carry over. Attending a conference needs approval from the line manager."),
    ]),
    ("IT Security and Equipment", [
        ("b", "Laptops are replaced every 36 months. Employees can choose between Linux, macOS and Windows."),
        ("b", "Passwords must have at least 14 characters, and multi-factor authentication is mandatory for all "
              "company accounts."),
        ("b", "A lost or stolen device must be reported within 2 hours to security@aurelia-robotics.example or "
              "through the incident hotline, extension 7777."),
        ("b", "USB storage devices are prohibited on the production-lab computers at the Port Alder plant."),
        ("b", "Customer data may never be stored on personal devices."),
        ("b", "Software may only be installed from the approved catalog. Exceptions require an IT ticket."),
    ]),
    ("Performance, Pay and Notice Periods", [
        ("b", "Performance reviews take place twice a year, in April and October, using a rating scale from 1 to 5."),
        ("b", "Promotions are decided once a year, in the October cycle."),
        ("b", "Bonus: employees can earn an annual bonus of up to 12% of base salary, depending on company targets."),
        ("b", "Stock options vest over 4 years with a 1-year cliff."),
        ("b", "Salary bands are reviewed every January."),
        ("b", "Probation lasts 6 months. After probation, the notice period is 2 months for both sides."),
    ]),
]

MANUAL = [
    ("Meridian HX-200 Water Pump: Overview and Specifications", [
        ("p", "The Meridian HX-200 is a self-priming centrifugal water pump for garden irrigation, rainwater "
              "transfer and light drainage. Please read this manual completely before first use and keep it for "
              "future reference."),
        ("h", "Technical data"),
        ("b", "Maximum flow rate: 120 litres per minute"),
        ("b", "Maximum head: 38 metres"),
        ("b", "Motor power: 1.1 kW, supply 230 V at 50 Hz"),
        ("b", "Weight: 14.5 kg"),
        ("b", "Maximum water temperature: 40 degrees Celsius"),
        ("b", "Housing material: cast aluminium, protection class IP55"),
    ]),
    ("Safety Warnings", [
        ("b", "Never run the pump dry. Running without water destroys the mechanical seal within minutes."),
        ("b", "Do not pump flammable or corrosive liquids such as petrol, solvents or acids."),
        ("b", "The pump can handle solid particles up to 4 mm in diameter. Larger particles block the impeller."),
        ("b", "Connect the pump only to a circuit protected by a residual-current device (RCD) rated at 30 mA."),
        ("b", "Always disconnect the power plug before any maintenance or cleaning."),
        ("b", "Keep children and pets away from the pump while it is running."),
        ("b", "Do not use the pump for the household drinking water supply."),
    ]),
    ("Installation", [
        ("b", "Place the pump on a flat, stable surface with at least 30 cm of free space around it for ventilation."),
        ("b", "Use a suction hose with an inner diameter of at least 32 mm. The maximum suction lift is 7 metres."),
        ("b", "The outlet connection is 25 mm (1 inch)."),
        ("b", "Before the first start, fill the pump housing with 1.5 litres of water (priming). "
              "Without priming the pump cannot draw water."),
        ("b", "Fit a check valve in the suction line and use a foot filter to keep debris out."),
    ]),
    ("Maintenance", [
        ("b", "Clean the inlet filter every 50 operating hours, or more often with dirty water."),
        ("b", "Replace the mechanical seal every 2,000 operating hours. Use the genuine seal kit MK-200."),
        ("b", "The bearings are sealed for life and must not be lubricated."),
        ("b", "Before winter, drain all water from the pump housing and store the pump in a frost-free room."),
        ("b", "An annual inspection by an authorised service partner is recommended."),
    ]),
    ("Troubleshooting and Warranty", [
        ("h", "Error codes"),
        ("b", "E1: dry run detected. The pump stops after 30 seconds. Fill the housing with water and restart."),
        ("b", "E2: motor overheating. The thermal cutoff stops the motor at 85 degrees Celsius. Let it cool down."),
        ("b", "E3: blocked impeller. Disconnect power and remove the blockage."),
        ("h", "Status LED"),
        ("p", "A steady green LED means the pump is running. A flashing blue LED means the pump is in pairing mode "
              "with the Meridian app."),
        ("h", "Warranty and support"),
        ("p", "The warranty is 24 months for consumer use and 12 months for commercial use. Proof of purchase is "
              "required. Support hotline: +00 800 555 0142."),
    ]),
]

if __name__ == "__main__":
    build("Aurelia_Robotics_Employee_Handbook.pdf", "Aurelia Robotics Ltd. Employee Handbook 2026", HANDBOOK)
    build("Meridian_HX-200_Pump_Manual.pdf", "Meridian HX-200 Owner's Manual", MANUAL)
