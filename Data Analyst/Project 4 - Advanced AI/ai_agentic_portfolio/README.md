# AI-Native Data Analyst Portfolio — Agentic AI, RAG & Voice AI

Three fully-built data analyst + AI projects, each built around a different 2026 AI trend (agentic AI, retrieval-augmented generation, voice/multimodal AI agents) and a different niche startup domain (AI customer support, healthtech, logistics) — chosen to read as directly relevant to the kind of AI-native startups currently hiring in the US, UK, and Singapore. Each project is a complete, independently runnable build: seeded synthetic data generation, real statistics/ML pipeline code, an executed Jupyter notebook (verified zero errors), a self-contained interactive Plotly dashboard, and a README case study with resume bullets and interview talking points. Every number quoted below is a real output of the code in this repository.

## The three projects

### 1. [ResolveIQ](1_resolveiq_cx_agents/) — Autonomous CX Agent Performance & Containment Analytics
**Trend: Agentic AI** (autonomous multi-step AI agents) · **Domain: AI customer support**
An AI support-agent platform's own self-reported success metric gets gamed by a policy change: official containment for one ticket category jumps +20.2pp (p<0.001) after a release, but the TRUE, reopen-adjusted containment rate is completely flat (p=0.98). Catching this requires validating an AI agent's self-report against real downstream customer behavior, not trusting the headline number.

### 2. [GroundTruth Clinical](2_groundtruth_clinical_rag/) — RAG Retrieval Quality & Clinical Hallucination Risk Analytics
**Trend: Retrieval-Augmented Generation (RAG)** · **Domain: HealthTech / clinical AI**
A clinical AI copilot's embedding-model upgrade improves retrieval quality for every query type in aggregate — except rare drug-combination queries (~20% of that category), where precision@k collapses (0.44 → 0.28, p=6.6×10⁻⁴³) and hallucination risk rises, invisible in the category-level average and compounded by physicians being less able to catch an error involving an unfamiliar interaction.

### 3. [RouteVoice](3_routevoice_logistics_voice/) — Logistics Voice AI Agent Call Quality & Containment Analytics
**Trend: Voice / Multimodal AI Agents** · **Domain: Logistics & supply chain**
A voice-AI model upgrade for a trucking-logistics dispatch platform looks like a clean nationwide win (+4.94pp containment) — until region-level analysis finds one region regressing (−6.75pp, p=3.6×10⁻⁶), driven by a measurable drop in transcription confidence consistent with a regional dialect coverage gap in the new model.

## The pattern that ties all three together

Each project is built around the same underlying analytical discipline, applied to a different 2026 AI product category: **an AI system's own top-line, self-reported, or aggregate metric can look completely fine while a real problem hides in a specific segment, and finding it requires deliberately validating the metric against ground truth or slicing by segment rather than trusting the headline number.** ResolveIQ validates an agent's self-reported outcome against real customer behavior; GroundTruth Clinical validates a category-level RAG quality metric against a segment; RouteVoice validates a nationwide rollout metric against a region. Each project also pairs its statistical finding with an LLM agent (perceive → act → reason → report) that applies an explicit, auditable rubric rather than an ad hoc judgment call — and, critically, each agent is tested on a case it should correctly NOT flag as a problem, not just the cases it should catch.

## On the shared LLM backend

All three projects use the same `shared/llm_client.py` abstraction used throughout this session's work: a `CachedClient` that replays real, hand-authored/data-grounded responses offline (every notebook runs end-to-end with zero API key required), and a `LiveClient` path that activates automatically the moment `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is present in the environment.

## Honest limitations, portfolio-wide

Every project states its specific simplifications in its own README: ResolveIQ's 48-hour reopen window is an imperfect proxy for "didn't actually work"; GroundTruth Clinical simulates retrieval outcomes rather than benchmarking against a real document corpus (and all clinical content is synthetic/illustrative, not real medical guidance); RouteVoice infers a training-data-coverage mechanism from confidence data rather than confirming it against the voice-model vendor's actual training composition. Every place a real experiment, a larger sample, or additional access would sharpen the answer is named explicitly as the next step, not glossed over.

## Running any project

```
cd <project_folder>/data && python3 generate_data.py
cd ../notebook && jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
cd ../dashboard && python3 build_dashboard.py   # writes index.html
```

Each project's `dashboard/index.html` is self-contained (Plotly inlined) and opens directly in a browser with no server required.
