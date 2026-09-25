import { Container, getContainer } from "@cloudflare/containers";

interface Env {
  STRATA_API: DurableObjectNamespace<StrataApi>;
  CORS_ORIGINS: string;
  OPENAI_API_KEY?: string;
}

// Runs the FastAPI server from ./Dockerfile (uvicorn on port 8000).
export class StrataApi extends Container<Env> {
  defaultPort = 8000;
  // Stop the container after 15 minutes without traffic to save cost.
  sleepAfter = "15m";
  // Container readiness check hits the API's own health route.
  pingEndpoint = "health";

  constructor(ctx: DurableObjectState<{}>, env: Env) {
    super(ctx, env);
    this.envVars = {
      CORS_ORIGINS: env.CORS_ORIGINS,
      // Optional: without it the API still returns risk results, just no LLM explanation.
      ...(env.OPENAI_API_KEY ? { OPENAI_API_KEY: env.OPENAI_API_KEY } : {}),
    };
  }
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // One shared instance: the model is stateless, so every request can use it.
    return getContainer(env.STRATA_API, "main").fetch(request);
  },
};
