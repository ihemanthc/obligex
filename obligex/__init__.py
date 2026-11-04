"""obligex — contract obligation extraction: schema, chunking, merge, eval."""

from .schema import (  # noqa: F401
    SCHEMA_VERSION,
    STUDENT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    BindingStrength,
    DocumentObligations,
    DocumentType,
    Modality,
    Money,
    Obligation,
    ObligationType,
    Party,
    PartyRole,
    Recurrence,
    SourceSpan,
    Timing,
    TimingType,
    WindowExtraction,
    assign_ids,
    build_messages,
    build_user_prompt,
    content_fingerprint,
    dumps_target,
    find_span,
    ground_obligations,
    guided_schema,
    parse_model_output,
)
from .chunking import Window, make_windows, split_blocks, windows_from_pages  # noqa: F401
from .merge import merge_document, reconcile_parties  # noqa: F401
from .evaluate import EvalReport, evaluate, match_obligations  # noqa: F401

__version__ = "0.1.0"
