# 🖥️ Real-Time Deepfake Detection via Frame-Level EfficientNet Ensemble (Desktop App)

## 💡 Application Overview

The **Desktop Application** allows users to upload videos, images, audio, and even input claims for news verification. These inputs are sent to a backend **Deep Learning model** via a lightweight **Ktor Client** for real-time analysis. The UI is built using **Jetpack Compose for Desktop**, offering a beautiful and modern experience.

Whether you're verifying a video, checking a suspicious audio clip, or fact-checking a piece of news, this app serves as a one-stop solution to detect DeepFakes and misinformation with ease.

<div align="center">
  <img src="https://github.com/user-attachments/assets/52154030-7b17-4b3e-90e4-36eddbbff5e5" />
</div>


---


## 🛠️ Technologies Used

- [Jetpack Compose](https://developer.android.com/compose)
- [KMP](https://kotlinlang.org/docs/multiplatform.html)
- [Kotlin](https://kotlinlang.org/)
- [Ktor Client](https://ktor.io/docs/create-client.html)
- [Kotlinx Serialization](https://kotlinlang.org/docs/serialization.html)
- [MVVM Architecture](https://developer.android.com/topic/architecture)
- [Kotlin Coroutines](https://kotlinlang.org/docs/coroutines-overview.html)
- [Clean Architecture](https://developer.android.com/topic/architecture)
- [Gradle Kotlin DSL](https://docs.gradle.org/current/userguide/kotlin_dsl.html)
- [File-Kit](https://github.com/vinceglb/FileKit)

---


## 🧠 Features

- 🎥 Upload and analyze **video files** in multiple formats: `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`
- 🖼️ Upload and analyze **image files**: `.jpg`, `.jpeg`, `.png`, `.bmp` with prediction and manipulation detection
- 🎵 Upload and verify **audio clips**: `.mp3`, `.wav`, `.ogg`, `.aac`, `.m4a` — supports fake voice detection
- 📰 Input **news claims** and get real-time similarity checks using reliable sources
- ⚡ Fast and responsive **real-time prediction engine** powered by a deep learning backend
- 🧩 Clean MVVM architecture with separation of concerns and flow-based state management
- 🎯 View detailed results including:
  - ✅ Prediction result (Fake / Real)
  - 📊 Confidence score
  - 🔍 Similarity insights (for news)
  - 🌐 Source links (for fact-checking)
- 💡 Smart error handling with toast-style feedback and custom error messages
- 🖥️ Built with **Jetpack Compose Desktop**: Modern, scalable, cross-platform UI framework
- 📱 **Responsive UI** adapts beautifully to different window sizes and high-DPI displays
- 🧰 FileKit-powered desktop file pickers for a clean UX


---

## Project Structure :

```
Main Package
│
├── 🗂️ data                # Network/API handling, DTOs, Repositories (implementation)
│
├── 🗂️ domain              # Use cases, Repository interfaces, Data models
│
├── 🗂️ presentation        # UI (Jetpack Compose), ViewModels, Navigation
│
└── 🗂️ constants           # API keys (Hidden)
```



