import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { admin } from '@/lib/api/admin';
import { useLanguage } from '@/contexts/LanguageContext';

interface DomainConfig {
  domain: string;
  dataSources: string[];
}

export default function DomainConfig() {
  const { t } = useLanguage();
  const [domains, setDomains] = useState<DomainConfig[]>([]);
  const [newDomain, setNewDomain] = useState('');
  const [error, setError] = useState('');

  const handleAddDomain = () => {
    if (newDomain) {
      setDomains([...domains, { domain: newDomain, dataSources: [] }]);
      setNewDomain('');
    }
  };

  const handleAddDataSource = (domainId: string, url: string) => {
    setDomains(domains.map(domain => {
      if (domain.domain === domainId) {
        return {
          ...domain,
          dataSources: [...domain.dataSources, url]
        };
      }
      return domain;
    }));
  };

  const handleRemoveDataSource = (domainId: string, index: number) => {
    setDomains(domains.map(domain => {
      if (domain.domain === domainId) {
        return {
          ...domain,
          dataSources: domain.dataSources.filter((_, i) => i !== index)
        };
      }
      return domain;
    }));
  };

  const handleSave = async (domain: DomainConfig) => {
    try {
      await admin.updateDomainConfig(domain);
      setError('');
    } catch (err) {
      setError(t('admin.domains.error'));
    }
  };

  return (
    <Card>
      <CardHeader>
        <h2 className="text-2xl font-bold">{t('admin.domains.title')}</h2>
        <p className="text-gray-500">{t('admin.domains.subtitle')}</p>
      </CardHeader>
      <CardContent>
        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="space-y-6">
          <div className="flex gap-2">
            <Input
              placeholder={t('admin.domains.enterDomain')}
              value={newDomain}
              onChange={(e) => setNewDomain(e.target.value)}
            />
            <Button onClick={handleAddDomain}>{t('admin.domains.add')}</Button>
          </div>

          <div className="space-y-4">
            {domains.map((domain) => (
              <DomainCard
                key={domain.domain}
                domain={domain}
                onAddDataSource={handleAddDataSource}
                onRemoveDataSource={handleRemoveDataSource}
                onSave={handleSave}
              />
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

interface DomainCardProps {
  domain: DomainConfig;
  onAddDataSource: (domainId: string, url: string) => void;
  onRemoveDataSource: (domainId: string, index: number) => void;
  onSave: (domain: DomainConfig) => void;
}

function DomainCard({ domain, onAddDataSource, onRemoveDataSource, onSave }: DomainCardProps) {
  const [newUrl, setNewUrl] = useState('');
  const { t } = useLanguage();

  return (
    <Card>
      <CardContent className="pt-6">
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <h4 className="font-medium">{domain.domain}</h4>
            <Button variant="outline" onClick={() => onSave(domain)}>
              {t('admin.domains.saveChanges')}
            </Button>
          </div>

          <div className="flex gap-2">
            <Input
              placeholder={t('admin.domains.enterUrl')}
              value={newUrl}
              onChange={(e) => setNewUrl(e.target.value)}
            />
            <Button
              variant="outline"
              onClick={() => {
                if (newUrl) {
                  onAddDataSource(domain.domain, newUrl);
                  setNewUrl('');
                }
              }}
            >
              {t('admin.domains.addUrl')}
            </Button>
          </div>

          <div className="space-y-2">
            {domain.dataSources.map((url, index) => (
              <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                <span className="text-sm">{url}</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onRemoveDataSource(domain.domain, index)}
                >
                  {t('admin.domains.remove')}
                </Button>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
