# Real-Time Deepfake Detection via Frame-Level EfficientNet  Ensemble and Client-Server Deployment

## Application Implementation :

The **Android application** enables users to upload videos directly from their device, send them to a backend server running a **deep learning model**, and receive predictions about the video's authenticity. Built using **Jetpack Compose** for a modern UI experience and Ktor for efficient network communication, the app provides users with **real-time** results and optional visual explanations using **Grad-CAM heatmaps**. This system bridges advanced machine learning with user-friendly mobile access, empowering users to verify video content on the go

<div align="center">
  <img src="https://github.com/user-attachments/assets/fccf7627-628e-4236-a687-8c491c0669dc"  />
</div>


## Technologies Used:
- [Jetpack Compose](https://developer.android.com/compose)
- [Kotlin](https://kotlinlang.org/)
- [CIO Engine](https://ktor.io/docs/client-engines.html)
- [Kotlinx Serialization (for JSON parsing)](https://kotlinlang.org/docs/serialization.html)
- [Android ViewModel (MVVM Architecture)](https://developer.android.com/topic/architecture)
- [Coroutine (for asynchronous tasks)](https://kotlinlang.org/docs/coroutines-overview.html)
- [Clean Architecture](https://developer.android.com/topic/architecture)
- [Splash API](https://developer.android.com/develop/ui/views/launch/splash-screen)
- [Dagger Hilt](https://dagger.dev/hilt/)
- [TypeSafe Navigation](https://developer.android.com/jetpack/androidx/releases/navigation)
- [Fancy Toast](https://github.com/Shashank02051997/FancyToast-Android)
- [Coil](https://coil-kt.github.io/coil/)
- [Lottie Animation](https://lottiefiles.com/)
- [ExoPlayer](https://developer.android.com/reference/androidx/media3/exoplayer/video/package-summary)
- [Ktor](https://ktor.io/)
- [Kapt](https://kotlinlang.org/docs/kapt.html)



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
├── 🗂️ di                  # Dependency Injection setup (e.g., Hilt modules)
│
└── 🗂️ constants           # API keys (Hidden)
```

## Features:
- Upload and analyze videos for DeepFake detection
- Real-time prediction results
- Displays prediction score and status (Fake/Real)
- Request Grad-CAM heatmap visualization for fake predictions
- Lightweight and clean modern UI using Jetpack Compose


