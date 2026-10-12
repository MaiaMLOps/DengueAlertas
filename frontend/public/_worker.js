export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Enrutar todas las llamadas /api/* hacia https://api.arkoiot.com/
    if (url.pathname.startsWith('/api/')) {
      const targetPath = url.pathname.replace(/^\/api/, '');
      const targetUrl = new URL(targetPath + url.search, 'https://api.arkoiot.com');

      const headers = new Headers(request.headers);
      
      // Inyectar Service Tokens de Cloudflare Zero Trust (configurados en Pages)
      if (env.CF_ACCESS_CLIENT_ID) {
        headers.set('CF-Access-Client-Id', env.CF_ACCESS_CLIENT_ID);
      }
      if (env.CF_ACCESS_CLIENT_SECRET) {
        headers.set('CF-Access-Client-Secret', env.CF_ACCESS_CLIENT_SECRET);
      }

      // Preservar el Origin del frontend para que Nginx enrute entre Staging y Prod
      const origin = request.headers.get('Origin') || url.origin;
      headers.set('Origin', origin);
      headers.set('Host', 'api.arkoiot.com');

      const proxyRequest = new Request(targetUrl.toString(), {
        method: request.method,
        headers: headers,
        body: request.method !== 'GET' && request.method !== 'HEAD' ? request.body : undefined,
        redirect: 'follow',
      });

      try {
        const response = await fetch(proxyRequest);
        return response;
      } catch (err) {
        return new Response(JSON.stringify({ error: 'Gateway Proxy Error', details: err.message }), {
          status: 502,
          headers: { 'Content-Type': 'application/json' },
        });
      }
    }

    // Servir assets estáticos de React / Vite
    return env.ASSETS.fetch(request);
  },
};
