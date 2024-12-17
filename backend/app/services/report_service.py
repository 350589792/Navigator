from sqlalchemy.orm import Session
from app.models.user import User
from app.crud import crud_report
from app.schemas.report import ReportCreate
from app.services import crawler_service, llm_service

async def generate_and_save_report(db: Session, user: User):
    """
    Generate and save a report for a user based on their domain preferences.
    """
    # Collect data from all user's domains
    all_data = []
    for domain in user.domains:
        data = await crawler_service.fetch_domain_data(domain)
        all_data.extend(data)

    # Generate report using LLM
    title, content, summary = await llm_service.generate_report(all_data)

    # Create report
    report_in = ReportCreate(
        user_id=user.id,
        title=title,
        content=content,
        summary=summary
    )

    return crud_report.create(db, obj_in=report_in)
