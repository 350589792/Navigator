from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from app.api import deps
from app.crud import crud_report
from app.schemas.report import Report, ReportCreate
from app.models.user import User
try:
    from app.services import report_service, email_service
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False
    print("Warning: Report and email services not available")

router = APIRouter()

@router.get("/", response_model=List[Report])
def get_reports(
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    Get all reports for current user.
    """
    return crud_report.get_user_reports(db, user_id=current_user.id)

@router.post("/generate")
async def generate_report(
    background_tasks: BackgroundTasks,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    Generate a new report for user's domains.
    """
    if not current_user.domains:
        raise HTTPException(
            status_code=400,
            detail="Please select at least one domain first"
        )

    # Create a basic report
    report_in = ReportCreate(
        user_id=current_user.id,
        title="Industry Report",
        content="Report content will be generated soon.",
        summary="Report is being processed."
    )
    report = crud_report.create(db, obj_in=report_in)

    # Try to generate report in background if service is available
    if SERVICES_AVAILABLE:
        try:
            background_tasks.add_task(
                report_service.generate_and_save_report,
                db=db,
                user=current_user
            )
            return {"status": "Report generation started", "report_id": report.id}
        except Exception as e:
            print(f"Report generation failed: {str(e)}")
            return {"status": "Basic report created", "report_id": report.id}

    return {"status": "Basic report created", "report_id": report.id}

@router.post("/send/{report_id}")
async def send_report(
    report_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    Send a specific report via email.
    """
    report = crud_report.get(db, id=report_id)
    if not report or report.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Report not found")

    if not SERVICES_AVAILABLE:
        return {"status": "Email service not available"}

    try:
        background_tasks.add_task(
            email_service.send_report,
            email=current_user.email,
            report=report
        )
        return {"status": "Email sending started"}
    except Exception as e:
        print(f"Email sending failed: {str(e)}")
        return {"status": "Email service error", "error": str(e)}
