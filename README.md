# Multi-Agent Decision Support Prototype

## Hi, I'm Eishaan.

This is a prototype I put together to explore a better way of making complex decisions in subsurface exploration. The goal was to build a system that's clear, reliable, and easy to trace back—something you can trust.

I built this version for IEW 2026 as a part of the team to demonstrate the core concepts.

### The Big Idea

Making the call on whether to drill for oil and gas is a huge, expensive decision. Usually, you have different teams—geologists, engineers, finance—all looking at their own piece of the puzzle. It can be tough to bring all that information together in a way that's consistent and easy to double-check later on.

My approach here is to use a set of small, specialised "agents." Each one has a single job, and they work together to build a complete picture.

### How It Works

The workflow is pretty straightforward:

1.  **You start with a "site package"**: This is just a folder with the data for a potential drill site—well logs (as a CSV), a seismic image (as a PNG), and some basic metadata.
2.  **The Agents Get to Work**: Four agents analyze the data from different angles:
    *   **Petrophysical Agent:** Looks at the well logs to figure out the rock type and how much space there is for oil or gas.
    *   **Seismic Agent:** Checks the seismic image for signs of a trap that could hold those resources.
    *   **Reservoir Agent:** Combines the findings from the first two to give an overall "reservoir quality" score.
    *   **Risk/ESG Agent:** Looks at the potential economics and any non-technical risks, like whether the site is in an environmentally sensitive area.
3.  **A Final Recommendation**: A final "Consensus" step looks at the results from all the agents and makes a simple, final call: **DRILL** or **HOLD**.

Everything is saved along the way as simple JSON files and plots, so you can see exactly how the system arrived at its conclusion.

### The Demo Scenarios

To show how this works in practice, I've included four different scenarios in the `demo_assets_advanced` folder. Each one is designed to tell a story and test a specific part of the system:

| Scenario                                | What It Shows                                                                                    | Expected Outcome    |
| --------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------- |
| `SCENARIO_01_PRIME_DRILL_TARGET`        | The ideal case, where everything looks good.                                                     | **DRILL**           |
| `SCENARIO_02_GEOLOGICAL_FAIL_SHALE`     | The seismic looks promising, but the rocks are bad. This shows the log analysis agent doing its job. | **HOLD**            |
| `SCENARIO_03_SEISMIC_FAIL_NO_TRAP`      | The rocks look okay, but the seismic shows no trap. This highlights a critical geological risk.       | **HOLD**            |
| `SCENARIO_04_ESG_VETO_SENSITIVE_AREA` | The geology and seismic are perfect, but it's in a protected area. This shows the ESG agent's veto. | **HOLD (ESG VETO)** |


### A Quick Note on Scope

Since this was built in a single day, I had to keep the scope tight. This demo is all about proving the workflow and the concept. So, to be clear, it doesn't include:

*   **Real machine learning models.** The "agents" are simple, rule-based functions for now.
*   **Industry-standard file formats** like LAS or SEG-Y. I used standard CSVs and PNGs to keep things simple and reliable for the demo.
*   **A big database for the chat.** The chat feature runs on a small, in-memory index of the artifacts generated for the session.

The idea is that this prototype provides a solid, trustworthy foundation that real ML models and data connectors could be plugged into later.

Thanks for taking a look. I'm happy to walk you through it and answer any questions.
