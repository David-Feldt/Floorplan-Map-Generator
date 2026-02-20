# Agentic Perception Demo

**Real-time visual decision-making for delivery robots using depth camera + local LLM**

---

## Concept

A live demo showing how a robot perceives its environment through an **Orbbec depth camera** and makes contextual decisions in real-time. A simulated robot (animated square) attempts to navigate past a human while responding intelligently to unexpected situations — all driven purely by visual input.

---

## What It Does

The robot has one goal: **get past you**. It watches the scene continuously, feeds perception data to **Gemma** (via Ollama), and chooses from four actions displayed directly on the robot square:

- `WAIT`
- `MOVE FORWARD`
- `MOVE BACK`
- `SPEAK` — text appears on the robot itself, no audio / communicate

Each decision shows a **one-sentence reason** rendered on or below the robot square.

---

## Demo Scenarios

Triggered naturally through visual perception — no manual input:

| Scenario | Expected Behavior |
|---|---|
| Clear path | Move forward |
| Person blocking | Wait — *"Human detected in path at close range"* |
| Person injured / on ground | Wait + Speak — *"Person appears to be on the ground, waiting for safety"* |
| Threatening gesture / object | Move back — *"Potential threat detected, retreating to safe distance"* |
| Liquid / obstacle on floor | Wait or Move back — *"Hazard on floor detected ahead"* |
| Person waving through | Move forward — *"Human signaling clear path"* |

---

## System Architecture

```
Orbbec Depth Camera
        ↓
 Perception Layer
 - Depth map + RGB feed
 - SAM (Segment Anything Model) — primary segmentation
   └ alternatives: OpenCV, YOLO, or generalized trained models
 - Person pose, proximity, floor plane, scene delta
        ↓
 Scene Description (structured JSON snapshot)
 "Person at 1.2m, crouched, arms raised, obstacle on floor left"
        ↓
 Gemma (local, via Ollama)
 System prompt: robot goal + action space + safety rules
        ↓
 Action + 1-sentence reason → rendered on robot square in Pygame
```

---

## Pygame UI

- **Left panel** — live depth feed (colorized) with SAM segmentation overlay
- **Center** — robot square animating current action (grows/shrinks on movement cycle)
  - Action label rendered **on the square**
  - 1-sentence Gemma reason rendered **below the square**
- **Right panel** — scrolling decision log (action + reason history)

---

## Tech Stack

| Component | Detail |
|---|---|
| Depth camera | Orbbec |
| Perception | SAM (primary), OpenCV / YOLO / custom models as alternatives |
| Local LLM | Gemma via Ollama |
| UI | Pygame |

---

## Key Design Decisions

- **Vision-only** — everything the LLM knows comes from the camera, no manual triggers
- **SAM for segmentation** — scene objects and people are segmented before being described to Gemma; swappable for OpenCV or YOLO depending on compute
- **Structured scene snapshots** — depth + segmentation data is compressed into a short text description before the LLM call, keeping latency manageable
- **Stateful context** — Gemma receives the last N snapshots so it can reason about change over time (e.g. someone *fell*, water *appeared*)
- **Decoupled loop rates** — UI runs at 30fps, LLM inference runs at 2–5fps

---

## Stretch Goals

- Multi-person scene handling
- Confidence score per action shown on UI
- Record + replay sessions
- Swap robot "personalities" via system prompt

---

## File Structure Example

```
/demo
  main.py           # main loop, pygame UI
  perception.py     # orbbec camera + SAM segmentation + scene parsing
  llm_agent.py      # gemma/ollama interface, prompt builder
  ui.py             # rendering, robot square animation
  scenarios.md      # static scene descriptions for testing without camera
  README.md
```