import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: './', // relative asset paths, so the build works under any URL (e.g. GitHub Pages /<repo>/)
})
