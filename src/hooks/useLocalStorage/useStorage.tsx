import React from "react";
function useStorage(defaultValue: string, key: string) {
    const [value, setValue] = React.useState(() => {
        const localValue = window.localStorage.getItem(key)
        return localValue !== null ? JSON.parse(localValue) : defaultValue
    })

    React.useEffect(() => {
        window.localStorage.setItem(key, JSON.stringify(value))
    }, [key, value])
    return [value, setValue]
}
export default useStorage;