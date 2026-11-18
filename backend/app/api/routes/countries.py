"""Country reference routes (seeded in Alembic migration 0002)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.country import Country
from app.schemas.stations import CountryRead

router = APIRouter(tags=["countries"])


@router.get("", response_model=list[CountryRead], summary="List supported countries")
def list_countries(db: Session = Depends(get_db)) -> list[CountryRead]:
    countries = db.scalars(select(Country).order_by(Country.name)).all()
    return [
        CountryRead(
            id=c.id,
            iso_code=c.iso_code,
            name=c.name,
            default_timezone=c.default_timezone,
        )
        for c in countries
    ]
