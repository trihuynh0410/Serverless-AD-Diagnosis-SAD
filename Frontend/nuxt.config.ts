import { defineNuxtConfig } from 'nuxt/config';
import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();

export default defineNuxtConfig({
  nitro: {
    preset: 'cloudflare-pages'
  },
  devtools: { enabled: true },
  modules: ['@nuxt/ui', 'nuxt-icon-tw'],

  // Make environment variables available on the client-side
  runtimeConfig: {
    public: {
      API_BASE_URL: process.env.API_BASE_URL,
    },
  },
  
});
