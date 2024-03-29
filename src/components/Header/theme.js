'use client'
import React from "react";
import useStorage from "@/hooks/useLocalStorage/useStorage";


function ThemeSwitcher() {
    const storageKey = 'theme-preference'
    const [theme, setTheme] = useStorage('light', storageKey)

    const switchTheme = () => {
        // flip current value
        setTheme((prev) => prev === 'light' ? 'dark' : 'light')

        setPreference()
    }

    /* const getColorPreference = () => {
        if (localStorage.getItem(storageKey)) {
            return localStorage.getItem(storageKey)
        }
        else
            return window.matchMedia('(prefers-color-scheme: dark)').matches
                ? 'dark'
                : 'light'
    } */

    const setPreference = () => {
        localStorage.setItem(storageKey, theme)
        reflectPreference()
    }

    const reflectPreference = () => {
        document.firstElementChild
            .setAttribute('data-theme', theme)

        document
            .querySelector('#theme-toggle')
            ?.setAttribute('aria-label', theme)
    }

    React.useEffect(() => {
        // set early so no page flashes / CSS is made aware
        reflectPreference()

        window.onload = () => {
            // set on load so screen readers can see latest value on the button
            reflectPreference()


        }

        // sync with system changes
        window
            .matchMedia('(prefers-color-scheme: dark)')
            .addEventListener('change', ({ matches: isDark }) => {
                setTheme((prev) => prev = isDark ? 'dark' : 'light')
                setPreference()
            })
    })






    return (
        <button onClick={switchTheme} className="theme-toggle" id="theme-toggle" title="Toggles light & dark" aria-label="auto" aria-live="polite">
            <svg className="sun-and-moon" aria-hidden="true" width="18" height="18" viewBox="0 0 24 24">
                <mask className="moon" id="moon-mask">
                    <rect x="0" y="0" width="100%" height="100%" fill="white" />
                    <circle cx="24" cy="10" r="6" fill="black" />
                </mask>
                <circle className="sun" cx="12" cy="12" r="6" mask="url(#moon-mask)" fill="currentColor" />
                <g className="sun-beams" stroke="currentColor">
                    <line x1="12" y1="1" x2="12" y2="3" />
                    <line x1="12" y1="21" x2="12" y2="23" />
                    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                    <line x1="1" y1="12" x2="3" y2="12" />
                    <line x1="21" y1="12" x2="23" y2="12" />
                    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                </g>
            </svg>
        </button>
    )
}
export default ThemeSwitcher;
