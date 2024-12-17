import { api } from './base';
import type { DomainPreference } from '@/types/api';

export const domains = {
  updateDomains: async (domainIds: number[]): Promise<DomainPreference> => {
    const response = await api.post<DomainPreference>('/domains/preferences', { domain_ids: domainIds });
    return response.data;
  }
};
