import type { Config } from 'tailwindcss'
import defaultTheme from 'tailwindcss/defaultTheme'

export default <Partial<Config>>{
  theme: {
    extend: {
      colors: {
        'blue': {
            '50': '#edf1ff',
            '100': '#dde6ff',
            '200': '#c3cfff',
            '300': '#9eafff',
            '400': '#7784ff',
            '500': '#575bfd',
            '600': '#4339f2',
            '700': '#392cd6',
            '800': '#2f27ac',
            '900': '#2b2788',
            '950': '#1b174f',
        },        
      }
    }
  }
}
