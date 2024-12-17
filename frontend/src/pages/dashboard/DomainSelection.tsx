import { useState } from 'react';
import { Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useLanguage } from '@/contexts/LanguageContext';
import { domains } from '@/lib/api/domains';

const AVAILABLE_DOMAINS = [
  { id: 1, label: { zh: '科技', en: 'Technology' } },
  { id: 2, label: { zh: '金融', en: 'Finance' } },
  { id: 3, label: { zh: '医疗', en: 'Medical' } },
  { id: 4, label: { zh: '人工智能', en: 'Artificial Intelligence' } },
];

export default function DomainSelection() {
  const { t, language } = useLanguage();
  const [selectedDomains, setSelectedDomains] = useState<number[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isSaved, setIsSaved] = useState(false);

  const toggleDomain = (domainId: number) => {
    setSelectedDomains(prev =>
      prev.includes(domainId)
        ? prev.filter(id => id !== domainId)
        : [...prev, domainId]
    );
    setIsSaved(false);
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setError(null);
    try {
      await domains.updateDomains(selectedDomains);
      setIsSaved(true);
    } catch (error) {
      console.error('Failed to update domains:', error);
      setError(t('dashboard.domains.error'));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <h2 className="text-2xl font-bold">{t('dashboard.domains.title')}</h2>
        <p className="text-gray-500">{t('dashboard.domains.subtitle')}</p>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {AVAILABLE_DOMAINS.map(domain => (
            <Button
              key={domain.id}
              variant={selectedDomains.includes(domain.id) ? 'default' : 'outline'}
              className="justify-start gap-2"
              onClick={() => toggleDomain(domain.id)}
            >
              {selectedDomains.includes(domain.id) && (
                <Check className="w-4 h-4" />
              )}
              {domain.label[language]}
            </Button>
          ))}
        </div>
        {error && (
          <Alert variant="destructive" className="mt-4">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        {isSaved && (
          <Alert className="mt-4">
            <AlertDescription>{t('dashboard.domains.saved')}</AlertDescription>
          </Alert>
        )}
        <Button
          onClick={handleSubmit}
          className="mt-6 w-full"
          disabled={isLoading}
        >
          {isLoading ? t('common.loading') : t('dashboard.domains.save')}
        </Button>
      </CardContent>
    </Card>
  );
}
