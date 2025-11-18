# ReaSyn25

**ReaSyn25** aims to create an accessible pipeline for generating
medical scenarios and Script Concordance Test (SCT) items. This project
is facilitated by the **Vanderbilt University Medical Center, Department
of Biomedical Informatics**.

The ultimate goal is to build a large benchmark of generated clinical
data---validated by clinicians and LLM-as-judges---to evaluate the
clinical reasoning capabilities of language models.

The pipeline is designed to be accessible for clinicians and researchers
with minimal technical background. Users interact with the system
primarily through the provided **IPYNB driver notebooks**.

------------------------------------------------------------------------

## Pipeline Overview

![Pipeline Figure](figures/ReaSyn%20Flow.png "Pipeline Example Figure")

------------------------------------------------------------------------

## Currently Implemented Models

-   **GPT-4.1**
-   **GPT-5** *(with GPT-5.1 support coming soon)*
-   **OpenAI o3**
-   **DeepSeek Chat 3.2**
-   **Claude Sonnet 4.5**
-   **Gemini 2.5 Pro** *(Gemini 3 support coming soon)*
-   **Claude Opus 4.1**
-   **Kimi K2 Thinking**

------------------------------------------------------------------------

## Setup Requirements

A `.env` file must be created containing your personal API keys for each
model's respective provider.

------------------------------------------------------------------------

## Project Status

This project is **incomplete** and **under active development**.