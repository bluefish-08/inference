'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import PageContainer from '@/components/ui/page-container';
import { Select } from '@/components/ui/select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { LOG_REFRESH_OPTIONS } from '@/constants/logs';
import { useI18n, type TFunc } from '@/contexts/i18n-context';
import request from '@/lib/request';
import { cn } from '@/lib/utils';

import { TimeRangePicker } from '../log-center/time-range-picker';
import type { TimeRangeValue } from '../log-center/types';

interface Summary {
  avg: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
}

interface GroupRow {
  key: string;
  count: number;
  share: number;
  errors: number;
  prompt_tokens: number;
  completion_tokens: number;
  models?: number;
  api_keys?: number;
  model_type?: string;
  user?: string;
  client_ip?: string;
  latency: Summary;
  ttft: Summary;
  tps: Summary;
  series: number[];
}

interface StatsResponse {
  total: number;
  errors: number;
  unique_users: number;
  unique_api_keys: number;
  unique_models: number;
  prompt_tokens: number;
  completion_tokens: number;
  latency: Summary;
  ttft: Summary;
  tps: Summary;
  interval: string;
  series: { ts: string; count: number; errors: number }[];
  by_model: GroupRow[];
  by_api_key: GroupRow[];
}

const EMPTY_SUMMARY: Summary = { avg: 0, p50: 0, p95: 0, p99: 0, max: 0 };

const EMPTY_STATS: StatsResponse = {
  total: 0,
  errors: 0,
  unique_users: 0,
  unique_api_keys: 0,
  unique_models: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  latency: EMPTY_SUMMARY,
  ttft: EMPTY_SUMMARY,
  tps: EMPTY_SUMMARY,
  interval: '1h',
  series: [],
  by_model: [],
  by_api_key: [],
};

const DEFAULT_TIME_RANGE: TimeRangeValue = { from: 'now-24h', to: 'now' };

// Distinct hues for the stacked series. Kept few on purpose: past this many
// models the stack stops being readable and the per-row sparklines carry it.
const SERIES_COLORS = [
  'hsl(221 83% 53%)',
  'hsl(174 84% 32%)',
  'hsl(262 83% 58%)',
  'hsl(32 95% 44%)',
  'hsl(199 89% 48%)',
  'hsl(330 81% 52%)',
];

const formatMs = (ms: number) =>
  ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
const formatCount = (value: number) => value.toLocaleString();
const dash = (value: number, render: (v: number) => string) => (value ? render(value) : '—');

function Metric({
  label,
  value,
  unit,
  hint,
  tone,
}: {
  label: string;
  value: string;
  unit?: string;
  hint?: string;
  tone?: 'danger';
}) {
  return (
    <div className="flex flex-col gap-0.5 px-4 py-3 first:pl-5 last:pr-5 md:border-l md:first:border-l-0">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div
        className={cn(
          'text-2xl font-semibold leading-tight tabular-nums',
          tone === 'danger' && 'text-destructive'
        )}
      >
        {value}
        {unit && <span className="ml-1 text-sm font-medium text-muted-foreground">{unit}</span>}
      </div>
      <div className="text-[11px] tabular-nums text-muted-foreground">{hint || ' '}</div>
    </div>
  );
}

function Sparkline({ values, color }: { values: number[]; color: string }) {
  const max = Math.max(...values, 1);
  const step = values.length > 1 ? 88 / (values.length - 1) : 88;
  const points = values
    .map((v, i) => `${(i * step).toFixed(1)},${(22 - (v / max) * 18).toFixed(1)}`)
    .join(' L');
  return (
    <svg width="88" height="24" viewBox="0 0 88 24" className="block" aria-hidden="true">
      <path d={`M${points}`} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

function StackedBars({
  series,
  rows,
  t,
}: {
  series: StatsResponse['series'];
  rows: GroupRow[];
  t: TFunc;
}) {
  const height = 168;
  const top = 16;
  const span = height - top;
  const max = Math.max(...series.map((point) => point.count), 1);
  const slot = series.length ? 1184 / series.length : 1184;
  const barWidth = Math.max(2, Math.min(32, slot - 6));
  // Hourly counts are discrete buckets, so they are drawn as bars rather than a
  // smoothed area, which would imply values between the buckets.
  const scale = (v: number) => (v === 0 ? 0 : Math.max(1.5, (v / max) * span));

  const ticks = series.length
    ? [
        0,
        Math.floor(series.length / 4),
        Math.floor(series.length / 2),
        Math.floor((series.length * 3) / 4),
        series.length - 1,
      ]
    : [];

  return (
    <svg width="1280" height="196" viewBox="0 0 1280 196" className="w-full">
      <g stroke="hsl(var(--border))" strokeWidth="1">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <line key={f} x1="48" y1={top + span * f} x2="1232" y2={top + span * f} />
        ))}
      </g>
      <g fill="hsl(var(--muted-foreground))" fontSize="10" textAnchor="end" className="font-mono">
        {[1, 0.75, 0.5, 0.25, 0].map((f) => (
          <text key={f} x="40" y={top + span * (1 - f) + 4}>
            {Math.round(max * f)}
          </text>
        ))}
      </g>

      {series.map((point, i) => {
        const x = 48 + i * slot + (slot - barWidth) / 2;
        let cursor = height;
        return (
          <g key={point.ts}>
            {rows.map((row, r) => {
              const value = row.series[i] || 0;
              const h = scale(value);
              if (!h) return null;
              cursor -= h;
              return (
                <rect
                  key={row.key}
                  x={x}
                  y={cursor}
                  width={barWidth}
                  height={h}
                  rx="1"
                  fill={SERIES_COLORS[r % SERIES_COLORS.length]}
                />
              );
            })}
          </g>
        );
      })}

      <path
        fill="none"
        stroke="hsl(var(--destructive))"
        strokeWidth="1.75"
        strokeLinejoin="round"
        d={series
          .map(
            (point, i) =>
              `${i === 0 ? 'M' : 'L'}${(48 + i * slot + slot / 2).toFixed(1)},${(height - scale(point.errors)).toFixed(1)}`
          )
          .join(' ')}
      />

      <g
        fill="hsl(var(--muted-foreground))"
        fontSize="10"
        textAnchor="middle"
        className="font-mono"
      >
        {ticks.map((i) => (
          <text key={i} x={48 + i * slot + slot / 2} y="188">
            {new Date(series[i].ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </text>
        ))}
      </g>
      {!series.length && (
        <text x="640" y="96" fill="hsl(var(--muted-foreground))" fontSize="12" textAnchor="middle">
          {t('usageStats.empty')}
        </text>
      )}
    </svg>
  );
}

function BreakdownTable({
  title,
  rows,
  keyLabel,
  memberLabel,
  memberField,
  colored,
  t,
}: {
  title: string;
  rows: GroupRow[];
  keyLabel: string;
  memberLabel: string;
  memberField: 'models' | 'api_keys';
  colored: boolean;
  t: TFunc;
}) {
  return (
    <div className="overflow-hidden rounded-lg border bg-card">
      <div className="border-b px-4 py-3 text-sm font-semibold">{title}</div>
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow className="[&_th]:text-[10px] [&_th]:font-semibold [&_th]:uppercase [&_th]:tracking-wider [&_th]:text-muted-foreground">
              <TableHead>{keyLabel}</TableHead>
              <TableHead className="w-28">{t('usageStats.trend')}</TableHead>
              <TableHead className="w-44 text-right">{t('usageStats.calls')}</TableHead>
              <TableHead className="w-20 text-right">{t('usageStats.errors')}</TableHead>
              <TableHead className="w-24 text-right">{t('usageStats.inputTokens')}</TableHead>
              <TableHead className="w-24 text-right">{t('usageStats.outputTokens')}</TableHead>
              <TableHead className="w-24 text-right">{memberLabel}</TableHead>
              <TableHead className="w-24 text-right">{t('usageStats.ttftP95')}</TableHead>
              <TableHead className="w-24 text-right">{t('usageStats.speedP50')}</TableHead>
              <TableHead className="w-24 text-right">{t('usageStats.latencyP95')}</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.length === 0 && (
              <TableRow>
                <TableCell colSpan={10} className="py-10 text-center text-muted-foreground">
                  {t('usageStats.empty')}
                </TableCell>
              </TableRow>
            )}
            {rows.map((row, index) => {
              const color = colored
                ? SERIES_COLORS[index % SERIES_COLORS.length]
                : 'hsl(var(--primary))';
              return (
                <TableRow key={row.key}>
                  <TableCell className="max-w-64">
                    <div className="flex min-w-0 items-center gap-2">
                      {colored && (
                        <span
                          className="size-2 shrink-0 rounded-sm"
                          style={{ background: color }}
                        />
                      )}
                      <span className="truncate font-medium">{row.key}</span>
                      {row.model_type && (
                        <span className="shrink-0 text-xs text-muted-foreground">
                          {row.model_type}
                        </span>
                      )}
                    </div>
                    {row.user && (
                      <div className="truncate text-xs text-muted-foreground">
                        {row.user}
                        {row.client_ip ? ` · ${row.client_ip}` : ''}
                      </div>
                    )}
                  </TableCell>
                  <TableCell>
                    <Sparkline values={row.series} color={color} />
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center justify-end gap-2.5">
                      <span className="h-1.5 w-16 shrink-0 overflow-hidden rounded-full bg-muted">
                        <span
                          className="block h-full rounded-full"
                          style={{
                            width: `${rows[0].count ? (row.count / rows[0].count) * 100 : 0}%`,
                            background: color,
                          }}
                        />
                      </span>
                      <span className="tabular-nums">{formatCount(row.count)}</span>
                      <span className="w-11 text-right text-xs tabular-nums text-muted-foreground">
                        {row.share}%
                      </span>
                    </div>
                  </TableCell>
                  <TableCell
                    className={cn(
                      'text-right tabular-nums',
                      row.errors > 0 ? 'text-destructive' : 'text-muted-foreground'
                    )}
                  >
                    {row.errors}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {dash(row.prompt_tokens, formatCount)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {dash(row.completion_tokens, formatCount)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{row[memberField] ?? 0}</TableCell>
                  <TableCell className="whitespace-nowrap text-right tabular-nums">
                    {dash(row.ttft.p95, formatMs)}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-right tabular-nums">
                    {dash(row.tps.p50, (v) => `${v} tok/s`)}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-right tabular-nums">
                    {dash(row.latency.p95, formatMs)}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

export default function UsageStats() {
  const { t } = useI18n();
  const [timeRange, setTimeRange] = useState<TimeRangeValue>(DEFAULT_TIME_RANGE);
  const [refreshInterval, setRefreshInterval] = useState(0);
  const [stats, setStats] = useState<StatsResponse>(EMPTY_STATS);
  const [loading, setLoading] = useState(true);
  const seqRef = useRef(0);

  const fetchStats = useCallback(
    async (silent = false) => {
      const seq = ++seqRef.current;
      if (!silent) setLoading(true);
      try {
        const params = new URLSearchParams({
          time_from: timeRange.from,
          time_to: timeRange.to,
          top_n: '10',
        });
        const data = await request.get<StatsResponse>('/v1/audit/stats?' + params.toString());
        if (seq === seqRef.current) setStats({ ...EMPTY_STATS, ...data });
      } catch {
        if (seq === seqRef.current && !silent) setStats(EMPTY_STATS);
      } finally {
        if (seq === seqRef.current) setLoading(false);
      }
    },
    [timeRange]
  );

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  useEffect(() => {
    if (!refreshInterval) return;
    const timer = setInterval(() => fetchStats(true), refreshInterval);
    return () => clearInterval(timer);
  }, [fetchStats, refreshInterval]);

  const errorRate = stats.total ? (stats.errors / stats.total) * 100 : 0;
  const stackedRows = useMemo(
    () => stats.by_model.slice(0, SERIES_COLORS.length),
    [stats.by_model]
  );

  return (
    <PageContainer
      title={t('usageStats.title')}
      subTitle={t('usageStats.description')}
      loading={loading}
      extraContent={
        <div className="flex items-center gap-2">
          <TimeRangePicker value={timeRange} onChange={setTimeRange} />
          <Select
            className="h-9 w-28"
            value={String(refreshInterval)}
            onChange={(value) => setRefreshInterval(Number(value))}
            options={LOG_REFRESH_OPTIONS.map((option) => ({
              label: t(option.labelKey),
              value: String(option.value),
            }))}
          />
          <Button variant="outline" size="icon" className="h-9 w-9" onClick={() => fetchStats()}>
            <RefreshCw className="size-4" />
          </Button>
        </div>
      }
    >
      <div className="flex flex-col gap-4">
        <div className="grid grid-cols-2 divide-y rounded-lg border bg-card md:grid-cols-4 md:divide-y-0 xl:grid-cols-7">
          <Metric label={t('usageStats.totalCalls')} value={formatCount(stats.total)} />
          <Metric
            label={t('usageStats.errorRate')}
            value={`${errorRate.toFixed(1)}%`}
            hint={`${stats.errors} / ${stats.total}`}
            tone={stats.errors > 0 ? 'danger' : undefined}
          />
          <Metric
            label={t('usageStats.activeKeys')}
            value={String(stats.unique_api_keys)}
            hint={t('usageStats.userCount', { count: stats.unique_users })}
          />
          <Metric label={t('usageStats.uniqueModels')} value={String(stats.unique_models)} />
          <Metric
            label={t('usageStats.outputTokens')}
            value={formatCount(stats.completion_tokens)}
            hint={`${t('usageStats.inputTokens')} ${formatCount(stats.prompt_tokens)}`}
          />
          <Metric
            label={t('usageStats.medianSpeed')}
            value={stats.tps.p50 ? String(stats.tps.p50) : '—'}
            unit={stats.tps.p50 ? 'tok/s' : undefined}
            hint={stats.tps.avg ? `${t('usageStats.average')} ${stats.tps.avg}` : undefined}
          />
          <Metric
            label={t('usageStats.ttftP95')}
            value={stats.ttft.p95 ? formatMs(stats.ttft.p95) : '—'}
            hint={stats.ttft.p50 ? `P50 ${formatMs(stats.ttft.p50)}` : undefined}
          />
        </div>

        <div className="rounded-lg border bg-card px-5 pb-2 pt-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-baseline gap-2.5">
              <span className="text-sm font-semibold">{t('usageStats.callVolumeByModel')}</span>
              <span className="text-xs text-muted-foreground">
                {t('usageStats.bucket', { interval: stats.interval })}
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
              {stackedRows.map((row, i) => (
                <span key={row.key} className="flex items-center gap-1.5">
                  <span
                    className="size-2.5 rounded-sm"
                    style={{ background: SERIES_COLORS[i % SERIES_COLORS.length] }}
                  />
                  {row.key}
                  <span className="tabular-nums">{formatCount(row.count)}</span>
                </span>
              ))}
              <span className="flex items-center gap-1.5">
                <span className="h-0.5 w-4 rounded-full bg-destructive" />
                {t('usageStats.errors')}
                <span className="tabular-nums">{stats.errors}</span>
              </span>
            </div>
          </div>
          <StackedBars series={stats.series} rows={stackedRows} t={t} />
        </div>

        <BreakdownTable
          title={t('usageStats.byModel')}
          rows={stats.by_model}
          keyLabel={t('usageStats.model')}
          memberLabel={t('usageStats.apiKeys')}
          memberField="api_keys"
          colored
          t={t}
        />
        <BreakdownTable
          title={t('usageStats.byApiKey')}
          rows={stats.by_api_key}
          keyLabel="API Key"
          memberLabel={t('usageStats.models')}
          memberField="models"
          colored={false}
          t={t}
        />
      </div>
    </PageContainer>
  );
}
