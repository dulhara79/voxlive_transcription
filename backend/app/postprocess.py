"""
Post-processing of FINAL segments only (never partials).

Default = light whitespace cleanup. To enable real LLM cleanup, set
ENABLE_POSTPROCESS=true and implement `_llm_clean`. Hard rules for the prompt:
  - PRESERVE the original languages and scripts. Do NOT translate.
  - Only fix punctuation, casing, spacing, obvious ASR slips.
  - Return the cleaned text and nothing else.
"""


class PostProcessor:
    async def process(self, text: str, language: str) -> str:
        return " ".join(text.split())

    async def _llm_clean(self, text: str, language: str) -> str:
        return text
