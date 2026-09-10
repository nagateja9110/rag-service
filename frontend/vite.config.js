import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // 5173 is Vite's default and is already in the backend's CORS allowlist
    // (see cors_allow_origins in config.local.yaml). Changing it here means
    // changing it there too, or the browser will block every request.
    port: 5173,
  },
})
