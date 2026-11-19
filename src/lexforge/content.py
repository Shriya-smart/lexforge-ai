"""Original educational challenges and examples grouped by contract topic."""

from dataclasses import dataclass

from .config import LABELS

DIFFICULTIES = ("Beginner", "Intermediate", "Advanced")


@dataclass(frozen=True)
class Check:
    description: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class Challenge:
    id: str
    label: str
    text: str
    checks: tuple[Check, Check]
    difficulty: str = ""


def entry(identifier, category, text, first, second, difficulty="") -> Challenge:
    return Challenge(identifier, LABELS[category], text,
                     (Check(first[0], first[1:]), Check(second[0], second[1:])), difficulty)


DRAFTS = (
    entry("d01", 0, "The receiving party shall keep confidential information",
          ("State a nondisclosure obligation", "confidential", "secret", "not disclose"),
          ("Name the party or group to whom disclosure is restricted", "third parties", "any person", "anyone", "public"),
          DIFFICULTIES[0]),
    entry("d02", 1, "Either party may terminate this agreement",
          ("Require written notice", "written notice", "notice in writing"),
          ("Give a notice period", r"\b(?:\d+|thirty|sixty|ten|fifteen|ninety)\s+(?:business\s+)?days?\b"),
          DIFFICULTIES[0]),
    entry("d03", 2, "This agreement shall be governed by",
          ("Refer to the applicable law", "laws?", "legal rules"),
          ("Name a jurisdiction", "delaware", "new york", "california", "england", "ontario", "singapore", "texas", "india"),
          DIFFICULTIES[0]),
    entry("d04", 3, "The supplier shall indemnify",
          ("Identify the protected party", "customer", "buyer", "purchaser", "client"),
          ("Identify a covered claim or loss", "claims?", "loss(?:es)?", "damages", "liabilit(?:y|ies)"),
          DIFFICULTIES[0]),
    entry("d05", 4, "The service provider's total liability shall not exceed",
          ("State a monetary cap", r"\$\s*\d", "fees (?:paid|payable)", "amount paid", "\\bUSD\\s*\\d"),
          ("Identify the basis or scope of the cap", "under this agreement", "preceding", "prior", "in aggregate", "all claims", "per claim"),
          DIFFICULTIES[0]),
    entry("d06", 0, "The recipient may disclose confidential information only",
          ("Specify a permitted recipient", "employees", "advisers", "advisors", "representatives", "counsel"),
          ("Limit disclosure by need or duty", "need to know", "need-to-know", "bound by", "confidentiality obligations"),
          DIFFICULTIES[1]),
    entry("d07", 1, "If a party materially breaches this agreement, the other party may",
          ("Identify termination as the remedy", "terminate", "end (?:this|the) agreement"),
          ("Provide a cure period", "(?:cure|remedy).{0,80}days?", "days?.{0,80}(?:cure|remedy)"),
          DIFFICULTIES[1]),
    entry("d08", 2, "The interpretation of this agreement shall be subject to",
          ("Select the law of a named jurisdiction", "laws? of (?:the state of )?(?:delaware|new york|california|england|ontario|singapore|texas|india)"),
          ("Address conflict-of-laws rules", "conflict[s -]+of[ -]+laws?", "choice[ -]+of[ -]+laws?"),
          DIFFICULTIES[1]),
    entry("d09", 3, "The vendor shall defend and indemnify the customer against",
          ("Identify third-party claims", "third[ -]party claims"),
          ("Identify the event that gives rise to a claim", "infringement", "negligence", "misconduct", "personal injury", "breach"),
          DIFFICULTIES[1]),
    entry("d10", 4, "Subject to the exceptions in this agreement, neither party shall be liable for",
          ("Name a type of excluded damages", "consequential", "indirect", "special damages", "lost profits"),
          ("Connect the exclusion to claims or notice of the loss", "even if advised", "regardless", "claims?", "foreseeable", "arising"),
          DIFFICULTIES[1]),
    entry("d11", 0, "If a court orders disclosure of confidential information, the recipient shall",
          ("Require notice where legally permitted", "notify", "give (?:prompt |written )?notice", "provide (?:prompt |written )?notice"),
          ("Limit disclosure to what is required", "minimum", "only.{0,80}(?:required|compelled)", "portion.{0,80}required"),
          DIFFICULTIES[2]),
    entry("d12", 1, "Upon termination of this agreement, the parties shall",
          ("Identify an accrued payment obligation", "(?:accrued|outstanding|unpaid).{0,40}(?:fees|amounts|payments)", "fees due"),
          ("Identify a duty that survives termination", "survive", "remain in effect", "continue to apply", "remain bound"),
          DIFFICULTIES[2]),
    entry("d13", 2, "The parties agree that any non-contractual obligations arising from this agreement",
          ("Select a named jurisdiction", "delaware", "new york", "california", "england", "ontario", "singapore", "texas", "india"),
          ("Expressly select substantive law", "governed by", "subject to the laws?", "substantive laws?", "laws? shall apply"),
          DIFFICULTIES[2]),
    entry("d14", 3, "As a condition of receiving indemnification, the protected party shall",
          ("Require notice of the claim", "notify", "notice"),
          ("Allocate control of defense or settlement", "control.{0,40}(?:defen[sc]e|settlement)", "consent.{0,40}settle", "assume.{0,40}defen[sc]e"),
          DIFFICULTIES[2]),
    entry("d15", 4, "The aggregate liability cap in this agreement shall not apply to",
          ("Identify an exception for misconduct", "fraud", "wil(?:l)?ful misconduct", "gross negligence"),
          ("Identify a separate exception", "indemn", "confidential", "data protection", "personal injury", "death"),
          DIFFICULTIES[2]),
)

GUESSES = (
    ("g01", LABELS[0], "The recipient may share the disclosing party’s trade secrets only with personnel who need them to perform this agreement."),
    ("g02", LABELS[1], "A party may end this agreement if the other party fails to cure a material breach within twenty days after written notice."),
    ("g03", LABELS[2], "The laws of Ontario govern this agreement and all claims arising from its performance."),
    ("g04", LABELS[3], "The vendor shall reimburse the customer for losses from third-party patent infringement claims caused by the vendor’s software."),
    ("g05", LABELS[4], "Neither party’s aggregate liability for breach may exceed the fees paid in the twelve months preceding the claim."),
)

REVISIONS = (
    entry("r01", 0, "The recipient will keep it secret.",
          ("Identify the protected information", "confidential information", "trade secrets?", "nonpublic information"),
          ("Describe who may or may not receive it", "third[ -]part", "employees", "authorized personnel", "representatives")),
    entry("r02", 1, "Either party can cancel soon.",
          ("Require written notice", "written notice", "notice in writing"),
          ("Specify a number of days", r"\b(?:\d+|thirty|sixty|ten|fifteen|ninety)\s+(?:business\s+)?days?\b")),
    entry("r03", 2, "Local law applies.",
          ("Identify an actual jurisdiction", "delaware", "new york", "california", "england", "ontario", "singapore", "texas", "india"),
          ("Tie the law to the agreement", "this agreement", "this contract")),
    entry("r04", 3, "Company A pays if something goes wrong.",
          ("Identify an indemnification obligation", "indemnif", "hold harmless", "reimburse"),
          ("Identify a covered claim and its source", "third[ -]party claims?", "infringement claims?", "negligence claims?")),
    entry("r05", 4, "Damages are limited to some fees.",
          ("State the liability cap", "amount (?:of )?fees", "fees (?:paid|payable)", r"\$\s*\d"),
          ("Specify the period or scope", "preceding", "prior", "calendar year", "per claim", "in aggregate", "under this agreement")),
)
