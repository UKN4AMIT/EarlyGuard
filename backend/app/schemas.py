from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["normal", "abnormal", "present", "notpresent", "yes", "no", "good", "poor"]

class KidneyPatientInput(BaseModel):
    """Patient measurements. Missing values are imputed by the saved training preprocessor."""
    model_config = ConfigDict(extra="forbid")

    age: Optional[float] = Field(None, ge=0, le=120)
    bp: Optional[float] = Field(None, ge=0, le=300)
    sg: Optional[float] = Field(None, ge=0.0, le=2.0)
    al: Optional[float] = Field(None, ge=0, le=10)
    su: Optional[float] = Field(None, ge=0, le=10)
    rbc: Optional[Literal["normal", "abnormal"]] = None
    pc: Optional[Literal["normal", "abnormal"]] = None
    pcc: Optional[Literal["present", "notpresent"]] = None
    ba: Optional[Literal["present", "notpresent"]] = None
    bgr: Optional[float] = Field(None, ge=0)
    bu: Optional[float] = Field(None, ge=0)
    sc: Optional[float] = Field(None, ge=0)
    sod: Optional[float] = Field(None, ge=0)
    pot: Optional[float] = Field(None, ge=0)
    hemo: Optional[float] = Field(None, ge=0)
    pcv: Optional[float] = Field(None, ge=0)
    wc: Optional[float] = Field(None, ge=0)
    rc: Optional[float] = Field(None, ge=0)
    htn: Optional[Literal["yes", "no"]] = None
    dm: Optional[Literal["yes", "no"]] = None
    cad: Optional[Literal["yes", "no"]] = None
    appet: Optional[Literal["good", "poor"]] = None
    pe: Optional[Literal["yes", "no"]] = None
    ane: Optional[Literal["yes", "no"]] = None

    @model_validator(mode="after")
    def require_at_least_one_value(self):
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("Provide at least one patient parameter.")
        return self


class KidneyPredictionResponse(BaseModel):
    prediction: Literal["CKD", "Not CKD"]
    predicted_class: int
    ckd_probability: float
    not_ckd_probability: float
    model: str
    disclaimer: str
