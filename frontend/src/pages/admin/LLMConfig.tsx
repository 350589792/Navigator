import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { admin } from '@/lib/api/admin';
import { useLanguage } from '@/contexts/LanguageContext';

interface LLMConfig {
  apiKey: string;
  baseUrl: string;
  model: string;
}

export default function LLMConfig() {
  const { t } = useLanguage();
  const [config, setConfig] = useState<LLMConfig>({ apiKey: '', baseUrl: '', model: '' });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleSave = async () => {
    try {
      await admin.updateLLMConfig(config);
      setSuccess(t('admin.llm.saved'));
      setError('');
    } catch (err) {
      setError(t('admin.llm.error'));
      setSuccess('');
    }
  };

  return (
    <Card>
      <CardHeader>
        <h2 className="text-2xl font-bold">{t('admin.llm.title')}</h2>
        <p className="text-gray-500">{t('admin.llm.subtitle')}</p>
      </CardHeader>
      <CardContent>
        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        {success && (
          <Alert className="mb-4">
            <AlertDescription>{success}</AlertDescription>
          </Alert>
        )}

        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">{t('admin.llm.apiKey')}</label>
            <Input
              type="password"
              placeholder={t('admin.llm.apiKeyPlaceholder')}
              value={config.apiKey}
              onChange={(e) => setConfig({ ...config, apiKey: e.target.value })}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">{t('admin.llm.baseUrl')}</label>
            <Input
              placeholder={t('admin.llm.baseUrlPlaceholder')}
              value={config.baseUrl}
              onChange={(e) => setConfig({ ...config, baseUrl: e.target.value })}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">{t('admin.llm.model')}</label>
            <Input
              placeholder={t('admin.llm.modelPlaceholder')}
              value={config.model}
              onChange={(e) => setConfig({ ...config, model: e.target.value })}
            />
          </div>

          <Button onClick={handleSave} className="w-full">
            {t('admin.llm.save')}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
