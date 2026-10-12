import { fileURLToPath, URL } from "node:url"
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, process.cwd(), '')

    return {
        plugins: [react(), tailwindcss()],
        resolve: {
            alias: {
                "@": fileURLToPath(new URL("./src", import.meta.url)),
            },
        },
        server: {
            proxy: {
                '/api': {
                    target: 'https://api.arkoiot.com',
                    changeOrigin: true,
                    rewrite: (path) => path.replace(/^\/api/, ''),
                    headers: {
                        'CF-Access-Client-Id': env.CF_ACCESS_CLIENT_ID,
                        'CF-Access-Client-Secret': env.CF_ACCESS_CLIENT_SECRET,
                    },
                },
            },
        },
    }
})