create table if not exists public.research_jobs (
  id text primary key,
  company_name text,
  country_code text,
  website_url text,
  status text not null default 'queued',
  progress integer not null default 0,
  message text,
  error text,
  result jsonb,
  report_storage_path text,
  created_at timestamptz default timezone('utc', now())
);

create index if not exists research_jobs_created_at_idx
  on public.research_jobs (created_at desc);

alter table public.research_jobs enable row level security;

create policy "service role full access on research_jobs"
on public.research_jobs
for all
using (auth.role() = 'service_role')
with check (auth.role() = 'service_role');
