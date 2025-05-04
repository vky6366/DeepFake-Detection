import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    
     <div className="max-w-md mx-auto bg-white shadow-lg rounded-xl p-6 space-y-4 font-sans">
            <h1 className="text-2xl font-bold text-center text-gray-800">
              🎬 Deepfake Video Detector
            </h1>



      <Header/>
      <Display/>
      <Loading/>
      <Result/>
      <Upload/>
      <VisualAnalysis/>
    </div>
  )
}

export default App
