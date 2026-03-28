from agents import OutputGuardrail, GuardrailFunctionOutput
from citation.verifier import CitationVerifier

verifier = CitationVerifier()


async def verify_citations(ctx, agent, output_text):
    """Output guardrail: verify all citations before returning to user."""
    verification = await verifier.verify_response(output_text)

    fabricated_count = len(verification.get("fabricated", []))
    mischaracterized_count = len(verification.get("mischaracterized", []))

    if fabricated_count > 0 or mischaracterized_count > 0:
        # Don't block - modify the response
        return GuardrailFunctionOutput(
            output_info={
                "clean_response": verification["clean_response"],
                "fabricated": verification["fabricated"],
                "mischaracterized": verification["mischaracterized"],
                "overruled": verification["overruled"],
            },
            tripwire_triggered=False,  # Don't block, just clean
        )

    return GuardrailFunctionOutput(
        output_info={"status": "all_citations_verified"},
        tripwire_triggered=False,
    )


citation_output_guardrail = OutputGuardrail(guardrail_function=verify_citations)
