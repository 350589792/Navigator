import { api } from './base';
import type { DomainPreference } from '@/types/api';

export const domains = {
  updateDomains: async (domains: string[]): Promise<DomainPreference> => {
    const response = await api.post<DomainPreference>('/users/preferences/domains', { domains });
    return response.data;
  }
};
