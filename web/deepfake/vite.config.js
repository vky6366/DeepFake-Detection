import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    tailwindcss(),
    react()],
    // server: {
    //   port: 5173,
    //   strictPort: false, // If true, Vite will throw an error if 5173 is busy. If false, it will try next ports like 5174, 5175, etc.
    // },

})

