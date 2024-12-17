import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Input } from '@/components/ui/input';
import { useLanguage } from '@/contexts/LanguageContext';
import * as reportsApi from '@/lib/api/reports';
import type { Report } from '@/types/api';

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
      setReportList(data);
      setError('');
    } catch (err) {
      setError(t('reports.fetchError'));
    } finally {
      setIsLoading(false);
    }
  };

  const filteredReports = reportList.filter(report =>
    report.domains.some(domain =>
      domain.toLowerCase().includes(filter.toLowerCase())
    ) ||
    report.date.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <h2 className="text-2xl font-bold">{t('reports.title')}</h2>
          <p className="text-gray-500">{t('reports.subtitle')}</p>
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
                placeholder={t('reports.filterPlaceholder')}
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="max-w-sm"
              />
              <Button onClick={fetchReports} disabled={isLoading}>
                {isLoading ? t('common.loading') : t('common.refresh')}
              </Button>
              <Button
                onClick={async () => {
                  setIsLoading(true);
                  try {
                    await reportsApi.reports.generateReport();
                    await fetchReports();
                  } catch (err) {
                    setError(t('reports.generateError'));
                  } finally {
                    setIsLoading(false);
                  }
                }}
                disabled={isLoading}
              >
                {t('reports.generateReport')}
              </Button>
            </div>

            {filteredReports.length === 0 ? (
              <p className="text-gray-500">{t('reports.noReports')}</p>
            ) : (
              <div className="grid gap-4">
                {filteredReports.map((report) => (
                  <Card key={report.id} className="cursor-pointer hover:bg-gray-50"
                        onClick={() => setSelectedReport(report)}>
                    <CardContent className="pt-6">
                      <div className="flex justify-between items-start">
                        <div>
                          <h3 className="font-medium">
                            {new Date(report.date).toLocaleDateString()}
                          </h3>
                          <p className="text-sm text-gray-500 mt-1">
                            {report.domains.join(', ')}
                          </p>
                        </div>
                      </div>
                      <p className="mt-2 text-sm line-clamp-2">{report.summary}</p>
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
                {t('reports.detailsTitle')} - {new Date(selectedReport.date).toLocaleDateString()}
              </h3>
              <Button variant="ghost" onClick={() => setSelectedReport(null)}>
                {t('common.close')}
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">{t('reports.domains')}</h4>
                <div className="flex gap-2 flex-wrap">
                  {selectedReport.domains.map((domain) => (
                    <span key={domain} className="px-2 py-1 bg-gray-100 rounded-full text-sm">
                      {domain}
                    </span>
                  ))}
                </div>
              </div>
              <div>
                <h4 className="font-medium mb-2">{t('reports.content')}</h4>
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
