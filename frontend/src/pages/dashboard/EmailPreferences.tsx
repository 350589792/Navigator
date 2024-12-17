import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useLanguage } from '@/contexts/LanguageContext';
import { user } from '@/lib/api/user';

const FREQUENCIES = [
  { value: 'daily', label: { zh: '每日', en: 'Daily' } },
  { value: 'weekly', label: { zh: '每周', en: 'Weekly' } },
  { value: 'monthly', label: { zh: '每月', en: 'Monthly' } },
  { value: 'workdays', label: { zh: '工作日', en: 'Workdays Only' } },
  { value: 'custom', label: { zh: '自定义', en: 'Custom Interval' } },
];

const TIMES = Array.from({ length: 24 }, (_, i) => ({
  value: `${i}:00`,
  label: { zh: `${i}:00`, en: `${i}:00` },
}));

export default function EmailPreferences() {
  const { t, language } = useLanguage();
  const [frequency, setFrequency] = useState('daily');
  const [time, setTime] = useState('09:00');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSaved, setIsSaved] = useState(false);

  const handleSubmit = async () => {
    setIsLoading(true);
    setError(null);
    try {
      await user.updateEmailPreferences({ frequency, time });
      setIsSaved(true);
    } catch (err) {
      console.error('Failed to update email preferences:', err);
      setError(t('dashboard.email.error'));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <h2 className="text-2xl font-bold">{t('dashboard.email.title')}</h2>
        <p className="text-gray-500">{t('dashboard.email.subtitle')}</p>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium">{t('dashboard.email.frequency')}</label>
            <Select value={frequency} onValueChange={setFrequency}>
              <SelectTrigger>
                <SelectValue placeholder={t('dashboard.email.selectFrequency')} />
              </SelectTrigger>
              <SelectContent>
                {FREQUENCIES.map(freq => (
                  <SelectItem key={freq.value} value={freq.value}>
                    {freq.label[language]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium">{t('dashboard.email.time')}</label>
            <Select value={time} onValueChange={setTime}>
              <SelectTrigger>
                <SelectValue placeholder={t('dashboard.email.selectTime')} />
              </SelectTrigger>
              <SelectContent>
                {TIMES.map(t => (
                  <SelectItem key={t.value} value={t.value}>
                    {t.label[language]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {isSaved && (
            <Alert>
              <AlertDescription>{t('dashboard.email.saved')}</AlertDescription>
            </Alert>
          )}

          <Button
            onClick={handleSubmit}
            className="w-full"
            disabled={isLoading}
          >
            {isLoading ? t('common.loading') : t('dashboard.email.save')}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
