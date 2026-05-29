import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'weather' },
  {
    path: 'weather',
    loadComponent: () =>
      import('./pages/weather-chat.component').then(
        (m) => m.WeatherChatComponent
      ),
  },
  {
    path: 'banking',
    loadComponent: () =>
      import('./pages/banking-interrupt.component').then(
        (m) => m.BankingInterruptComponent
      ),
  },
  {
    path: 'background',
    loadComponent: () =>
      import('./pages/background-agent.component').then(
        (m) => m.BackgroundAgentComponent
      ),
  },
];
