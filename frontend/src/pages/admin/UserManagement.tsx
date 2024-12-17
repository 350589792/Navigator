import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { admin } from '@/lib/api/admin';
import { useLanguage } from '@/contexts/LanguageContext';
import type { User } from '@/types/api';

export default function UserManagement() {
  const { t } = useLanguage();
  const [users, setUsers] = useState<User[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchUsers = async () => {
    setIsLoading(true);
    try {
      const users = await admin.getUsers();
      setUsers([...users]);
      setError('');
    } catch (err) {
      setError(t('admin.users.fetchError'));
    } finally {
      setIsLoading(false);
    }
  };

  const updateUserStatus = async (userId: string, status: 'active' | 'disabled') => {
    try {
      await admin.updateUser(userId, { status });
      await fetchUsers();
    } catch (err) {
      setError(t('admin.users.updateError'));
    }
  };

  return (
    <Card>
      <CardHeader>
        <h2 className="text-2xl font-bold">{t('admin.users.title')}</h2>
        <p className="text-gray-500">{t('admin.users.subtitle')}</p>
      </CardHeader>
      <CardContent>
        {error && (
          <Alert variant="destructive" className="mb-4">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <Input
              placeholder={t('admin.users.searchPlaceholder')}
              className="max-w-sm"
              onChange={() => {
                // TODO: Implement search functionality
              }}
            />
            <Button onClick={fetchUsers} disabled={isLoading}>
              {isLoading ? t('common.loading') : t('admin.users.refresh')}
            </Button>
          </div>
          <div className="divide-y">
            {users.map((user) => (
              <div key={user.id} className="py-4 flex justify-between items-center">
                <div>
                  <p className="font-medium">{user.name}</p>
                  <p className="text-sm text-gray-500">{user.email}</p>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant={user.status === 'active' ? 'destructive' : 'default'}
                    onClick={() => updateUserStatus(user.id, user.status === 'active' ? 'disabled' : 'active')}
                  >
                    {user.status === 'active' ? t('admin.users.disable') : t('admin.users.enable')}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
