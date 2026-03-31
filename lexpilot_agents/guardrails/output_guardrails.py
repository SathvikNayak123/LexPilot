from agents import OutputGuardrail, GuardrailFunctionOutput
from citation.verifier import CitationVerifier

verifier = CitationVerifier()


async def verify_citations(ctx, agent, output_text):
    """Output guardrail: verify all citations before returning to user.
    Trips the guardrail if fabricated citations are detected, forcing
    the agent to regenerate without them."""
    verification = await verifier.verify_response(output_text)

    fabricated_count = len(verification.get("fabricated", []))

    if fabricated_count > 0:
        fabricated_list = ", ".join(
            item["citation"] for item in verification["fabricated"]
        )
        return GuardrailFunctionOutput(
            output_info={
                "fabricated": verification["fabricated"],
                "message": f"Response contained {fabricated_count} fabricated citation(s): "
                           f"{fabricated_list}. These could not be verified in the index.",
            },
            tripwire_triggered=True,
        )

    return GuardrailFunctionOutput(
        output_info={"status": "all_citations_verified"},
        tripwire_triggered=False,
    )


citation_output_guardrail = OutputGuardrail(guardrail_function=verify_citations)
