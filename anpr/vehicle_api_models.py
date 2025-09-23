"""
Pydantic models for Norwegian Vehicle Data API
Based on the OpenAPI specification from Statens Vegvesen
"""

from typing import Optional, List
from datetime import datetime, date
from pydantic import BaseModel
from enum import Enum


class KodeverkType(BaseModel):
    """Code type from the Norwegian vehicle registry"""

    kodeBeskrivelse: Optional[str] = None
    kodeNavn: Optional[str] = None
    kodeTypeId: Optional[str] = None
    kodeVerdi: Optional[str] = None
    tidligereKodeVerdi: Optional[List[str]] = None


class KjennemerkeKategori(str, Enum):
    """License plate categories"""

    KJORETOY = "KJORETOY"
    NORMAL = "NORMAL"
    PERSONLIG = "PERSONLIG"
    PROVE = "PROVE"


class Adresse(BaseModel):
    """Address information"""

    adresselinje1: Optional[str] = None
    adresselinje2: Optional[str] = None
    adresselinje3: Optional[str] = None
    kommunenavn: Optional[str] = None
    kommunenummer: Optional[str] = None
    land: Optional[str] = None
    landkode: Optional[str] = None
    postnummer: Optional[str] = None
    poststed: Optional[str] = None


class PersonnavnMedFodselsdato(BaseModel):
    """Person name with birth date"""

    etternavn: Optional[str] = None
    fodselsdato: Optional[date] = None
    fornavn: Optional[str] = None
    mellomnavn: Optional[str] = None


class Enhet(BaseModel):
    """Organization unit"""

    organisasjonsnavn: Optional[str] = None
    organisasjonsnummer: Optional[str] = None


class PersonEnhetBegrenset(BaseModel):
    """Limited person/entity information"""

    adresse: Optional[Adresse] = None
    enhet: Optional[Enhet] = None
    fomTidspunkt: Optional[datetime] = None
    person: Optional[PersonnavnMedFodselsdato] = None
    tilTidspunkt: Optional[datetime] = None


class EierskapBegrenset(BaseModel):
    """Limited ownership information"""

    eier: Optional[PersonEnhetBegrenset] = None
    leasingtaker: Optional[PersonEnhetBegrenset] = None
    medeier: Optional[PersonEnhetBegrenset] = None
    underenhet: Optional[PersonEnhetBegrenset] = None
    vedtakstidspunkt: Optional[datetime] = None


class Kjennemerke(BaseModel):
    """License plate information"""

    fomTidspunkt: Optional[datetime] = None
    kjennemerke: Optional[str] = None
    kjennemerkekategori: Optional[KjennemerkeKategori] = None
    kjennemerketype: Optional[KodeverkType] = None
    tilTidspunkt: Optional[datetime] = None


class KjoretoyIdentitetBegrenset(BaseModel):
    """Limited vehicle identity"""

    kjennemerke: Optional[str] = None
    understellsnummer: Optional[str] = None
    kuid: Optional[str] = None


class Forstegangsregistrering(BaseModel):
    """First-time registration"""

    registrertForstegangNorgeDato: Optional[date] = None


class Registrering(BaseModel):
    """Vehicle registration information"""

    fomTidspunkt: Optional[datetime] = None
    kjoringensArt: Optional[KodeverkType] = None
    neringskode: Optional[str] = None
    neringskodeBeskrivelse: Optional[str] = None
    registreringsstatus: Optional[KodeverkType] = None
    registrertForstegangPaEierskap: Optional[datetime] = None
    tilTidspunkt: Optional[datetime] = None
    avregistrertSidenDato: Optional[datetime] = None


class PeriodiskKjoretoyKontroll(BaseModel):
    """Periodic vehicle inspection"""

    kontrollfrist: Optional[date] = None
    sistGodkjent: Optional[date] = None


class Merke(BaseModel):
    """Vehicle brand"""

    merke: Optional[str] = None
    merkeKode: Optional[str] = None


class Fabrikant(BaseModel):
    """Manufacturer information"""

    fabrikantAdresse: Optional[str] = None
    fabrikantNavn: Optional[str] = None
    fabrikantRepresentantAdresse: Optional[str] = None
    fabrikantRepresentantNavn: Optional[str] = None


class Generelt(BaseModel):
    """General vehicle information"""

    fabrikant: Optional[List[Fabrikant]] = None
    ferdigbyggetEllerEndretSomFolger: Optional[str] = None
    handelsbetegnelse: Optional[List[str]] = None
    merke: Optional[List[Merke]] = None
    tekniskKode: Optional[KodeverkType] = None
    tekniskUnderkode: Optional[KodeverkType] = None
    typebetegnelse: Optional[str] = None
    unntakFra: Optional[str] = None


class Dimensjoner(BaseModel):
    """Vehicle dimensions"""

    bredde: Optional[int] = None
    hoyde: Optional[int] = None
    lengde: Optional[int] = None
    lengdeInnvendigLasteplan: Optional[int] = None
    maksimalBredde: Optional[int] = None
    maksimalHoyde: Optional[int] = None
    maksimalLengde: Optional[int] = None
    maksimalLengdeInnvendigLasteplan: Optional[int] = None


class Vekter(BaseModel):
    """Vehicle weights"""

    egenvekt: Optional[int] = None
    egenvektMaksimum: Optional[int] = None
    egenvektMinimum: Optional[int] = None
    egenvektTilhengerkopling: Optional[int] = None
    frontOgHjulVekter: Optional[str] = None
    nyttelast: Optional[int] = None
    tekniskTillattForhoyetTotalvekt: Optional[int] = None
    tekniskTillattTotalvekt: Optional[int] = None
    tekniskTillattTotalvektVeg: Optional[int] = None
    tekniskTillattVektPahengsvogn: Optional[int] = None
    tekniskTillattVektSemitilhenger: Optional[int] = None
    tillattHjulLastSidevogn: Optional[int] = None
    tillattTaklast: Optional[int] = None
    tillattTilhengervektMedBrems: Optional[int] = None
    tillattTilhengervektUtenBrems: Optional[int] = None
    tillattTotalvekt: Optional[int] = None
    tillattVektSlepevogn: Optional[int] = None
    tillattVertikalKoplingslast: Optional[int] = None
    tillattVogntogvekt: Optional[int] = None
    tillattVogntogvektVeg: Optional[int] = None


class Drivstoff(BaseModel):
    """Fuel information"""

    drivstoffKode: Optional[KodeverkType] = None
    effektVektForhold: Optional[float] = None
    maksEffektPrTime: Optional[float] = None
    maksNettoEffekt: Optional[float] = None
    maksNettoEffektVedOmdreiningstallMin1: Optional[int] = None
    maksNettoEffektVedOmdreiningstallMin1Maks: Optional[int] = None
    maksOmdreining: Optional[int] = None
    spenning: Optional[float] = None
    tomgangsOmdreiningstall: Optional[int] = None


class Motor(BaseModel):
    """Engine information"""

    antallSylindre: Optional[int] = None
    arbeidsprinsipp: Optional[KodeverkType] = None
    avgassResirkulering: Optional[bool] = None
    blandingsDrivstoff: Optional[str] = None
    drivstoff: Optional[List[Drivstoff]] = None
    fabrikant: Optional[str] = None
    fordampningsutslippKontrollSystem: Optional[bool] = None
    katalysator: Optional[bool] = None
    kjolesystem: Optional[str] = None
    ladeluftkjoler: Optional[bool] = None
    luftInnsproytning: Optional[bool] = None
    motorKode: Optional[str] = None
    motornummer: Optional[str] = None
    oksygenSensor: Optional[bool] = None
    overladet: Optional[bool] = None
    partikkelfilterMotor: Optional[bool] = None
    slagvolum: Optional[int] = None
    sylinderArrangement: Optional[KodeverkType] = None


class MotorOgDrivverk(BaseModel):
    """Engine and drivetrain"""

    antallGir: Optional[int] = None
    antallGirBakover: Optional[int] = None
    effektKraftuttakKW: Optional[int] = None
    girPlassering: Optional[str] = None
    girkassetype: Optional[KodeverkType] = None
    giroverforngsType: Optional[str] = None
    hybridElektriskKjoretoy: Optional[bool] = None
    hybridKategori: Optional[KodeverkType] = None
    maksimumHastighet: Optional[List[int]] = None
    maksimumHastighetMalt: Optional[List[int]] = None
    motor: Optional[List[Motor]] = None
    obd: Optional[bool] = None
    totalUtvekslingHoyesteGir: Optional[float] = None
    utelukkenyElektriskDrift: Optional[bool] = None


class TekniskeData(BaseModel):
    """Technical data"""

    generelt: Optional[Generelt] = None
    dimensjoner: Optional[Dimensjoner] = None
    vekter: Optional[Vekter] = None
    motorOgDrivverk: Optional[MotorOgDrivverk] = None
    # Note: Additional technical data fields can be added as needed


class Kjoretoymerknad(BaseModel):
    """Vehicle note"""

    merknad: Optional[str] = None
    merknadtypeKode: Optional[str] = None


class TekniskGodkjenning(BaseModel):
    """Technical approval"""

    godkjenningsId: Optional[str] = None
    godkjenningsundertype: Optional[KodeverkType] = None
    gyldigFraDato: Optional[date] = None
    gyldigFraDatoTid: Optional[datetime] = None


class Godkjenning(BaseModel):
    """Vehicle approval"""

    kjoretoymerknad: Optional[List[Kjoretoymerknad]] = None
    tekniskGodkjenning: Optional[TekniskGodkjenning] = None


class EnkeltOppslagKjoretoydata(BaseModel):
    """Single lookup vehicle data"""

    kjoretoyId: Optional[KjoretoyIdentitetBegrenset] = None
    forstegangsregistrering: Optional[Forstegangsregistrering] = None
    kjennemerke: Optional[List[Kjennemerke]] = None
    registrering: Optional[Registrering] = None
    godkjenning: Optional[Godkjenning] = None
    periodiskKjoretoyKontroll: Optional[PeriodiskKjoretoyKontroll] = None


class KjoretoydataResponse(BaseModel):
    """Vehicle data response"""

    feilmelding: Optional[str] = None
    kjoretoydataListe: Optional[List[EnkeltOppslagKjoretoydata]] = None


# Error response models
class APIError(BaseModel):
    """API Error response"""

    message: str
    status_code: int
    details: Optional[str] = None
