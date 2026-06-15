import {
  Bar,
  BarChart,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';

import RecommendationCard from '../../components/cards/RecommendationCard';
import ChartCard from '../../components/charts/ChartCard';
import SectionHeader from '../../components/common/SectionHeader';
import DataTable from '../../components/tables/DataTable';
import type { DashboardDataBundle } from '../../types';
import { formatCurrency, formatPercent } from '../../utils/format';

interface SegmentationProps {
  data: DashboardDataBundle;
}

const colors = ['#2563eb', '#0ea5e9', '#10b981', '#f59e0b', '#ef4444'];

const Segmentation = ({ data }: SegmentationProps) => {
  const totalSegmentCustomers = data.segmentation.segmentDistribution.reduce(
    (acc, row) => acc + Number(row.customers || 0),
    0
  );

  const segmentRows = data.segmentation.segmentDistribution.map((row) => ({
    segment: String(row.name),
    customers: Number(row.customers || 0)
  }));

  const toShare = (count: number) =>
    `${((count / Math.max(totalSegmentCustomers, 1)) * 100).toFixed(1)}%`;

  const topCustomerColumns = [
    { key: 'customerId', label: 'Customer ID' },
    { key: 'state', label: 'State' },
    { key: 'segment', label: 'Segment' },
    { key: 'clv', label: 'CLV', render: (row: any) => formatCurrency(row.clv) }
  ];

  return (
    <section className="space-y-5">
      <SectionHeader
        title="Customer Segmentation for Demand Generation"
        subtitle="VALUE SEGMENTS"
        question="Which customer groups should receive retention, upsell, or controlled servicing actions?"
        takeaway="Segment-level visibility converts model output into clear action playbooks for account and marketing teams."
      />

      <div className="grid gap-4">
        <ChartCard
          title="Segment Distribution"
          subtitle="How customers are grouped"
          helperText="Shows segment share across High Value, Growth, Low Value, and Loss Making cohorts."
        >
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={data.segmentation.segmentDistribution} dataKey="customers" nameKey="name" innerRadius={60} outerRadius={105}>
                {data.segmentation.segmentDistribution.map((_, index) => (
                  <Cell key={index} fill={colors[index % colors.length]} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value: number) => [
                  `${Number(value).toLocaleString()} (${toShare(Number(value))})`,
                  'Customers'
                ]}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-soft dark:border-slate-800 dark:bg-slate-900">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Segment Share Calculation
          </h3>
          <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
            Share (%) = customers in segment / total customers in selected filters.
          </p>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="bg-slate-100 dark:bg-slate-800">
                <tr>
                  <th className="px-2 py-1 text-left">Segment</th>
                  <th className="px-2 py-1 text-right">Customers</th>
                  <th className="px-2 py-1 text-right">Share</th>
                </tr>
              </thead>
              <tbody>
                {segmentRows.map((row) => (
                  <tr key={row.segment} className="border-t border-slate-200 dark:border-slate-700">
                    <td className="px-2 py-1">{row.segment}</td>
                    <td className="px-2 py-1 text-right">{row.customers.toLocaleString()}</td>
                    <td className="px-2 py-1 text-right">{toShare(row.customers)}</td>
                  </tr>
                ))}
                <tr className="border-t border-slate-300 font-semibold dark:border-slate-600">
                  <td className="px-2 py-1">Total</td>
                  <td className="px-2 py-1 text-right">{totalSegmentCustomers.toLocaleString()}</td>
                  <td className="px-2 py-1 text-right">100.0%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </article>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <ChartCard
          title="Segment-wise Profit"
          subtitle="Profitability by segment"
          helperText="Compares profitability quality across each strategic customer segment."
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.segmentation.segmentProfit}>
              <XAxis dataKey="segment" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="avgProfit" fill="#10b981" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Segment-wise Renewal"
          subtitle="Retention quality by segment"
          helperText="Renewal trends indicate where long-term value is likely to persist or decay."
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.segmentation.segmentRenewal}>
              <XAxis dataKey="segment" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip formatter={(value: number) => formatPercent(Number(value))} />
              <Bar dataKey="renewalRate" fill="#0ea5e9" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <div className="grid gap-4">
        <article className="space-y-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-soft dark:border-slate-800 dark:bg-slate-900">
          <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">Top Customers</h3>
          <DataTable columns={topCustomerColumns as any} rows={data.segmentation.topCustomers as any} />
        </article>

          
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {data.segmentation.actionSummary.map((item) => (
          <RecommendationCard key={item.title} item={item} />
        ))}
      </div>
    </section>
  );
};

export default Segmentation;
