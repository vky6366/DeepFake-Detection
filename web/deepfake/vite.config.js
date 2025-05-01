import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    tailwindcss(),
    react()],
    
    //  server: {
    //   port: 5000,
    //   host: true,
      // proxy: {
      //   '/upload': 'http://localhost:5000',
      //   '/get_3rd_frame': 'http://localhost:5000',
      //   '/gradcam': 'http://localhost:5000',
      //   '/facial_analysis': 'http://localhost:5000'
      // }
    // }

})
