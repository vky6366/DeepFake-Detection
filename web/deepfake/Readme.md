# Real-Time Deepfake Detection via Frame-Level EfficientNet Ensemble and Client-Server Deployment

## Website Implementation :
The Web application enables users to upload videos directly from their device, send them to a backend server running a deep learning model, and receive predictions about the video's authenticity. Built using **TailwindCss** for better UI experience and **Reactjs** for efficient network communication, the web provides users with *real-time results and optional visual explanations using Grad-CAM heatmaps. This system bridges advanced machine learning with user-friendly web access, empowering users to verify video content on the go

![WebNavigation](https://github.com/user-attachments/assets/9f62b4af-7e16-4ffc-8095-b9d0e0433c70)

## Technologies Used:
- [React+vite](https://vite.dev/guide/)
- [Tailwindcss with vite](https://tailwindcss.com/)
- [Html](https://developer.mozilla.org/en-US/docs/Glossary/HTML5)
- [Axios](https://axios-http.com/docs/intro)

## Features:
- Upload and analyze videos for DeepFake detection
- Real-time prediction results
- Displays prediction score and status (Fake/Real)
-  Grad-CAM heatmap visualization for fake predictions
- Lightweight and clean UI using Tailwind Css


 ## 📦 Requirements

 ### React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

### Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [typescript-eslint](https://typescript-eslint.io) in your project.
 
### Installation of the react :
```react
npm create vite@latest
````

### Installtion  for tailwindcss :
```react
npm install tailwindcss @tailwindcss/vite
```

### Installtion of axios :
```react
npm install axios
```
