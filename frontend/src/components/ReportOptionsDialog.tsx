import { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { useLanguage } from "@/contexts/LanguageContext";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";

const AVAILABLE_DOMAINS = [
  { id: 1, label: { zh: '科技', en: 'Technology' } },
  { id: 2, label: { zh: '金融', en: 'Finance' } },
  { id: 3, label: { zh: '医疗', en: 'Medical' } },
  { id: 4, label: { zh: '人工智能', en: 'Artificial Intelligence' } },
];

interface ReportOptionsDialogProps {
  onGenerate: (options: {
    format: 'text' | 'html';
    domains: string[];
    preferences: {
      keywords: boolean;
      trends: boolean;
      keyPoints: boolean;
    };
  }) => Promise<void>;
  disabled?: boolean;
}

export function ReportOptionsDialog({ onGenerate, disabled }: ReportOptionsDialogProps) {
  const { t, language } = useLanguage();
  const [format, setFormat] = useState<'text' | 'html'>('text');
  const [selectedDomains, setSelectedDomains] = useState<string[]>([]);
  const [preferences, setPreferences] = useState({
    keywords: true,
    trends: true,
    keyPoints: true
  });
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const toggleDomain = (domainId: string) => {
    setSelectedDomains(prev =>
      prev.includes(domainId)
        ? prev.filter(id => id !== domainId)
        : [...prev, domainId]
    );
  };

  const handleGenerate = async () => {
    setIsLoading(true);
    try {
      await onGenerate({
        format,
        domains: selectedDomains,
        preferences
      });
      setIsOpen(false);
    } catch (error) {
      console.error('Failed to generate report:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button disabled={disabled}>{t('dashboard.reports.generateReport')}</Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{t('dashboard.reports.options.title')}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <Label>{t('dashboard.reports.options.format')}</Label>
            <Select value={format} onValueChange={(value: 'text' | 'html') => setFormat(value)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="text">{t('dashboard.reports.options.formatText')}</SelectItem>
                <SelectItem value="html">{t('dashboard.reports.options.formatHtml')}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label>{t('dashboard.domains.title')}</Label>
            <div className="grid grid-cols-2 gap-2">
              {AVAILABLE_DOMAINS.map(domain => (
                <Button
                  key={domain.id}
                  variant={selectedDomains.includes(String(domain.id)) ? 'default' : 'outline'}
                  className="justify-start"
                  onClick={() => toggleDomain(String(domain.id))}
                >
                  {domain.label[language]}
                </Button>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <Label>{t('dashboard.reports.options.preferences')}</Label>
            <Card className="p-4 space-y-3">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="keywords"
                  checked={preferences.keywords}
                  onCheckedChange={(checked) =>
                    setPreferences(prev => ({ ...prev, keywords: checked as boolean }))
                  }
                />
                <Label htmlFor="keywords">{t('dashboard.reports.options.keywords')}</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="trends"
                  checked={preferences.trends}
                  onCheckedChange={(checked) =>
                    setPreferences(prev => ({ ...prev, trends: checked as boolean }))
                  }
                />
                <Label htmlFor="trends">{t('dashboard.reports.options.trends')}</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="keyPoints"
                  checked={preferences.keyPoints}
                  onCheckedChange={(checked) =>
                    setPreferences(prev => ({ ...prev, keyPoints: checked as boolean }))
                  }
                />
                <Label htmlFor="keyPoints">{t('dashboard.reports.options.keyPoints')}</Label>
              </div>
            </Card>
          </div>

          <div className="flex justify-end space-x-2">
            <Button
              variant="outline"
              onClick={() => setIsOpen(false)}
            >
              {t('common.cancel')}
            </Button>
            <Button
              onClick={handleGenerate}
              disabled={isLoading}
            >
              {isLoading ? t('common.loading') : t('dashboard.reports.options.generate')}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
