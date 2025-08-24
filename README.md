# Context Awareness Text Augmented System

## Project Overview

The **Context Awareness Text Augmented System** is an open-source project developed under the guidance of **Professor Junxiao Shen** at the *University of Bristol*'s **BIG Lab**. The system uses **HoloLens 2** as a data collection platform and enables **Real-Time communication** of multimodal data (video, audio, eye-tracking) via a **Peer-to-Peer (P2P) connection** between **XR** and **PC systems**. This project aims to capture and process user behavior in an immersive environment and generate textual descriptions of these actions using AI models. 🤖🎮

## Key Features

- **Real-Time Multimodal Data Communication**: Supports the transmission of video, audio, and eye-tracking data in real-time between **HoloLens 2** and a PC system via a stable **P2P connection**. 🌐
- **Data to Text AI Models**: Utilizes AI-based models to convert raw multimodal data into text representations of user behavior. 🧠➡️📝
- **Multithreading for Parallel Data Handling**: Implements multithreading to ensure efficient data processing and timestamp synchronization for accurate communication. 💻⚙️
- **Question-Answering System**: Uses **LangChain** to create an AI-powered workflow for generating behavior-based question-answering systems, helping users recall daily activities. ❓🤖

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/username/context-awareness-text-augmented-system.git
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the **HoloLens 2** device for data collection and ensure the **P2P connection** is active between the device and the PC. 🔌

## Usage

### 1. Establishing Connection

   - Ensure both **HoloLens 2** and the PC are connected via the **P2P connection**. 🌐🔗

### 2. Data Collection

   - Use **HoloLens 2** to collect video, audio, and eye-tracking data. 🎥🎧👀

### 3. AI Workflow

   - The data will be processed by AI models to generate text versions of user actions. 💬
   - Use the generated text to interact with the **Question-Answering System** for daily behavior tracking. 📅📝

## Contribution

Contributions are welcome! To contribute, fork this repository, make your changes, and submit a pull request. 🔄💡

## License

This project is licensed under the MIT License 📄
