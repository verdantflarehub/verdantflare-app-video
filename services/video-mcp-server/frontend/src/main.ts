import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import '../dashboard.css'
import '../studio-theme.css'
import './theme/market.css'

createApp(App).use(createPinia()).mount('#app')
