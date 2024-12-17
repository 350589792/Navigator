import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Input } from '@/components/ui/input';
import { admin } from '@/lib/api/admin';
import { useLanguage } from '@/contexts/LanguageContext';
import type { Log } from '@/types/api';

export default function LogViewer() {
  const { t } = useLanguage();
  const [logs, setLogs] = useState<Log[]>([]);
  const [filter, setFilter] = useState('');
  const [crawlerFrequency, setCrawlerFrequency] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const response = await admin.getLogs();
      setLogs(response.data);
      setError('');
    } catch (err) {
      setError(t('admin.logs.error'));
    } finally {
      setLoading(false);
    }
  };

  const updateCrawlerFrequency = async () => {
    try {
      await admin.updateCrawlerFrequency(crawlerFrequency);
      setError('');
    } catch (err) {
      setError(t('admin.logs.crawler.error'));
    }
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  const filteredLogs = logs.filter(log =>
    log.message.toLowerCase().includes(filter.toLowerCase()) ||
    log.level.toLowerCase().includes(filter.toLowerCase()) ||
    (log.metadata && Object.values(log.metadata).some(value =>
      value.toString().toLowerCase().includes(filter.toLowerCase())))
  );

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <h2 className="text-2xl font-bold">{t('admin.logs.crawler.title')}</h2>
          <p className="text-gray-500">{t('admin.logs.crawler.subtitle')}</p>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Input
              placeholder={t('admin.logs.crawler.frequencyPlaceholder')}
              value={crawlerFrequency}
              onChange={(e) => setCrawlerFrequency(e.target.value)}
            />
            <Button onClick={updateCrawlerFrequency}>
              {t('admin.logs.crawler.update')}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <h2 className="text-2xl font-bold">{t('admin.logs.title')}</h2>
          <p className="text-gray-500">{t('admin.logs.subtitle')}</p>
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
                placeholder={t('admin.logs.filter')}
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="max-w-sm"
              />
              <Button onClick={fetchLogs} disabled={loading}>
                {loading ? t('admin.logs.loading') : t('admin.logs.refresh')}
              </Button>
            </div>

            <div className="border rounded-lg overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-muted">
                      <th className="px-4 py-2 text-left">{t('admin.logs.timestamp')}</th>
                      <th className="px-4 py-2 text-left">{t('admin.logs.level')}</th>
                      <th className="px-4 py-2 text-left">{t('admin.logs.message')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredLogs.map((log) => (
                      <tr key={log.id} className="border-t">
                        <td className="px-4 py-2 text-sm">{new Date(log.timestamp).toLocaleString()}</td>
                        <td className="px-4 py-2">
                          <span className={`inline-block px-2 py-1 rounded text-xs ${
                            log.level === 'error' ? 'bg-red-100 text-red-800' :
                            log.level === 'warning' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-green-100 text-green-800'
                          }`}>
                            {log.level}
                          </span>
                        </td>
                        <td className="px-4 py-2 text-sm">{log.message}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
