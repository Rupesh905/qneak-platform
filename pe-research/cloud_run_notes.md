Cloud Run setup

1. Build this folder as the deploy root.
2. Set environment variables in Cloud Run:
   - GOOGLE_API_KEY
   - EXA_API_KEY
   - TINYFISH_API_KEY
   - COMPANIES_HOUSE_API_KEY
   - SUPABASE_URL
   - SUPABASE_SERVICE_ROLE_KEY
   - SUPABASE_STORAGE_BUCKET
   - SUPABASE_JOBS_TABLE
   - ALLOWED_ORIGINS
3. In Supabase:
   - Run `supabase_schema.sql`
   - Create a private storage bucket named `research-reports`
4. Point your dashboard API base URL to this Cloud Run service or to a custom domain like `api.qneak.com`.
5. Keep the service role key only in Cloud Run, never in frontend code.
