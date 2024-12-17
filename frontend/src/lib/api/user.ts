import { api } from './base';

interface EmailPreferences {
  frequency: string;
  time: string;
}

export const user = {
  updateEmailPreferences: async (preferences: EmailPreferences): Promise<void> => {
    await api.post('/api/v1/users/preferences/email', preferences);
  }
};

export default user;
