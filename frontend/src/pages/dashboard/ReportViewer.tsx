import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Input } from '@/components/ui/input';
import { useLanguage } from '@/contexts/LanguageContext';
import * as reportsApi from '@/lib/api/reports';
import type { Report } from '@/types/api';
import { ReportOptionsDialog } from '@/components/ReportOptionsDialog';

export default function ReportViewer() {
  const { t } = useLanguage();
  const [reportList, setReportList] = useState<Report[]>([]);
  const [selectedReport, setSelectedReport] = useState<Report | null>(null);
  const [filter, setFilter] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchReports();
  }, []);

  const fetchReports = async () => {
    setIsLoading(true);
    try {
      const data = await reportsApi.reports.getReports();
      if (Array.isArray(data)) {
        setReportList(data);
        setError('');
      } else {
        setReportList([]);
        setError(t('dashboard.reports.fetchError'));
      }
    } catch (err) {
      setReportList([]);
      setError(t('dashboard.reports.fetchError'));
    } finally {
      setIsLoading(false);
    }
  };

  const filteredReports = reportList.filter(report =>
    filter === '' ? true : (
      (report.domains?.some(domain =>
        domain.toLowerCase().includes(filter.toLowerCase())
      ) || false) ||
      (report.date?.toLowerCase().includes(filter.toLowerCase()) || false)
    )
  );

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <h2 className="text-2xl font-bold">{t('dashboard.reports.title')}</h2>
          <p className="text-gray-500">{t('dashboard.reports.subtitle')}</p>
        </CardHeader>
        <CardContent>
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          <div className="space-y-4">
            <div className="flex gap-2">
              <Input
                placeholder={t('dashboard.reports.filterPlaceholder')}
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="max-w-sm"
              />
              <Button onClick={fetchReports} disabled={isLoading}>
                {isLoading ? t('common.loading') : t('dashboard.reports.refresh')}
              </Button>
              <ReportOptionsDialog
                onGenerate={async (options) => {
                  setIsLoading(true);
                  try {
                    const report = await reportsApi.reports.generateReport(options);
                    if (!report) {
                      setError(t('dashboard.reports.generateError'));
                    } else {
                      await fetchReports();
                      setError('');
                    }
                  } catch (err) {
                    setError(t('dashboard.reports.generateError'));
                  } finally {
                    setIsLoading(false);
                  }
                }}
                disabled={isLoading}
              />
            </div>

            {!reportList.length ? (
              <p className="text-gray-500">{t('dashboard.reports.noReports')}</p>
            ) : filteredReports.length === 0 ? (
              <p className="text-gray-500">{t('dashboard.reports.noMatchingReports')}</p>
            ) : (
              <div className="grid gap-4">
                {filteredReports.map((report) => (
                  <Card key={report.id} className="cursor-pointer hover:bg-gray-50"
                        onClick={() => setSelectedReport(report)}>
                    <CardContent className="pt-6">
                      <div className="flex justify-between items-start">
                        <div>
                          <h3 className="font-medium">
                            {report.date ? new Date(report.date).toLocaleDateString() : t('dashboard.reports.noDate')}
                          </h3>
                          <p className="text-sm text-gray-500 mt-1">
                            {Array.isArray(report.domains) && report.domains.length > 0
                              ? report.domains.join(', ')
                              : t('dashboard.reports.noDomains')}
                          </p>
                        </div>
                        {report.status && (
                          <span className={`text-sm px-2 py-1 rounded-full ${
                            report.status === 'processing' ? 'bg-blue-100 text-blue-700' :
                            report.status === 'completed' ? 'bg-green-100 text-green-700' :
                            'bg-red-100 text-red-700'
                          }`}>
                            {t(`dashboard.reports.${report.status}`)}
                          </span>
                        )}
                      </div>
                      <p className="mt-2 text-sm line-clamp-2">
                        {report.status === 'processing'
                          ? t('dashboard.reports.processing')
                          : report.status === 'error'
                            ? report.error || t('dashboard.reports.error')
                            : report.summary || t('dashboard.reports.noSummary')}
                      </p>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {selectedReport && (
        <Card>
          <CardHeader>
            <div className="flex justify-between items-center">
              <h3 className="text-xl font-bold">
                {t('dashboard.reports.detailsTitle')}
                {selectedReport.date && ` - ${new Date(selectedReport.date).toLocaleDateString()}`}
              </h3>
              <Button variant="ghost" onClick={() => setSelectedReport(null)}>
                {t('common.close')}
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">{t('dashboard.reports.domains')}</h4>
                <div className="flex gap-2 flex-wrap">
                  {Array.isArray(selectedReport.domains) && selectedReport.domains.length > 0
                    ? selectedReport.domains.map((domain) => (
                      <span key={domain} className="px-2 py-1 bg-gray-100 rounded-full text-sm">
                        {domain}
                      </span>
                    ))
                    : <span className="text-gray-500">{t('dashboard.reports.noDomains')}</span>}
                </div>
              </div>
              <div>
                <h4 className="font-medium mb-2">{t('dashboard.reports.content')}</h4>
                <div className="prose max-w-none">
                  {selectedReport.content}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
