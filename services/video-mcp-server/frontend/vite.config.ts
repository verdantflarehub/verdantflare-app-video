import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  base: '/dashboard/frontend/',
  build: { outDir: 'dist', emptyOutDir: true },
})
